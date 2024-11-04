import argparse
import os
import time
import multiprocessing
from typing import Tuple, Optional

import inference
import numpy as np
import stitching
# Do not import TensorFlow globally to avoid conflicts with CUDA_VISIBLE_DEVICES
# import tensorflow.compat.v1 as tf
import pandas as pd
from tqdm import tqdm
from distutils.util import strtobool


def preprocessing_worker(file_queue, preprocess_queue, args):
    # Restrict TensorFlow to CPU in this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs from TensorFlow
    import tensorflow.compat.v1 as tf  # Re-import TensorFlow after changing env

    preprocess_graph = tf.Graph()
    with tf.Session(graph=preprocess_graph) as sess:
        while True:
            item = file_queue.get()
            if item is None:
                file_queue.task_done()
                break  # Exit signal received
            index, row = item
            input_wav_fp = row["input_fp"]
            output_wav_fp = row["output_fp"]

            output_dir = os.path.dirname(output_wav_fp)
            os.makedirs(output_dir, exist_ok=True)

            try:
                hop_size_in_samples = args.block_size_in_samples // 2
                with preprocess_graph.as_default():
                    input_wav, sample_rate = inference.read_wav_file(
                        input_wav_fp, args.input_channels, args.scale_input)
                    input_wav = tf.transpose(input_wav)  # shape: [mics, samples]
                    input_len = tf.shape(input_wav)[-1]
                    input_wav = tf.pad(input_wav, [[0, 0], [hop_size_in_samples, 0]])
                    input_blocks = tf.signal.frame(
                        input_wav,
                        args.block_size_in_samples,
                        hop_size_in_samples,
                        pad_end=True
                    )
                    input_blocks *= stitching.get_window(args.window_type, args.block_size_in_samples)
                    input_blocks = tf.transpose(input_blocks, (1, 0, 2))

                input_blocks_np, input_len_np, sample_rate_np = sess.run(
                    [input_blocks, input_len, sample_rate])

                if sample_rate_np != args.sample_rate:
                    raise ValueError(f"Sample rate mismatch: expected {args.sample_rate}, got {sample_rate_np}")

                preprocess_queue.put((index, input_blocks_np, input_len_np, output_wav_fp))
                print(f"[Preprocessing] Completed for {input_wav_fp}")
            except Exception as e:
                print(f"[Preprocessing] Error processing {input_wav_fp}: {e}")
            finally:
                file_queue.task_done()


def model_inference_worker(preprocess_queue, postprocess_queue, args, use_gpu):
    if not use_gpu:
        # Restrict TensorFlow to CPU in this worker
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs from TensorFlow
    # Import TensorFlow after setting CUDA_VISIBLE_DEVICES
    import tensorflow.compat.v1 as tf  # Import TensorFlow

    # Initialize the model inside the worker
    model_graph_filename = os.path.join(args.model_dir, 'inference.meta')
    separation_model = inference.SeparationModel(
        args.checkpoint, model_graph_filename, args.input_tensor,
        args.output_tensor)
    while True:
        item = preprocess_queue.get()
        if item is None:
            preprocess_queue.task_done()
            break  # Exit signal received
        index, input_blocks_np, input_len_np, output_wav_fp = item
        try:
            output_blocks = []
            for i in range(input_blocks_np.shape[0]):
                # Run inference
                output = separation_model.separate(input_blocks_np[i])
                output_blocks.append(output)
            output_blocks_np = np.stack(output_blocks, axis=0)
            postprocess_queue.put((index, output_blocks_np, input_len_np, output_wav_fp))
            print(f"[Model Inference] Completed for {output_wav_fp}")
        except Exception as e:
            print(f"[Model Inference] Error processing {output_wav_fp}: {e}")
        finally:
            preprocess_queue.task_done()


def postprocessing_worker(postprocess_queue, args):
    # Restrict TensorFlow to CPU in this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs from TensorFlow
    import tensorflow.compat.v1 as tf  # Re-import TensorFlow after changing env

    postprocess_graph = tf.Graph()
    with tf.Session(graph=postprocess_graph) as sess:
        while True:
            item = postprocess_queue.get()
            if item is None:
                postprocess_queue.task_done()
                break  # Exit signal received
            index, output_blocks_np, input_len_np, output_wav_fp = item
            try:
                with postprocess_graph.as_default():
                    num_samples_in_block = output_blocks_np.shape[-1]
                    num_sources = output_blocks_np.shape[1]
                    hop_samples = num_samples_in_block // 2

                    output_blocks_placeholder = tf.placeholder(
                        tf.float32, shape=(None, num_sources, num_samples_in_block), name='output_blocks_placeholder')
                    input_len_placeholder = tf.placeholder(tf.int32, shape=(), name='input_len_placeholder')

                    output_blocks = output_blocks_placeholder
                    window = stitching.get_window(args.window_type, num_samples_in_block)

                    if args.permutation_invariant:
                        output_blocks = stitching.sequentially_resolve_permutation(
                            output_blocks, window)

                    output_blocks = tf.transpose(output_blocks, (1, 0, 2))
                    output_blocks *= window
                    output_wavs = tf.signal.overlap_and_add(output_blocks, hop_samples)
                    output_wavs = tf.transpose(output_wavs)
                    output_wavs = output_wavs[hop_samples: input_len_placeholder + hop_samples, :]

                    write_output_ops = inference.write_wav_file(
                        output_wav_fp, output_wavs, sample_rate=args.sample_rate,
                        num_channels=num_sources,
                        output_channels=args.output_channels,
                        write_outputs_separately=args.write_outputs_separately,
                        channel_name='source')

                    sess.run(write_output_ops,
                             feed_dict={output_blocks_placeholder: output_blocks_np,
                                        input_len_placeholder: input_len_np})
                print(f"[Post-processing] Completed for {output_wav_fp}")
            except Exception as e:
                print(f"[Post-processing] Error processing {output_wav_fp}: {e}")
            finally:
                postprocess_queue.task_done()


def process_chunk(chunk, args):
    manager = multiprocessing.Manager()
    file_queue = multiprocessing.JoinableQueue()
    preprocess_queue = multiprocessing.JoinableQueue(maxsize=args.queue_size)
    postprocess_queue = multiprocessing.JoinableQueue(maxsize=args.queue_size)

    # Start preprocessing workers
    preprocess_processes = []
    for _ in range(args.num_preprocess_workers):
        p = multiprocessing.Process(target=preprocessing_worker, args=(file_queue, preprocess_queue, args))
        p.start()
        preprocess_processes.append(p)

    # Start model inference workers
    model_processes = []

    # Start the first model inference worker with GPU
    p = multiprocessing.Process(target=model_inference_worker, args=(preprocess_queue, postprocess_queue, args, True))
    p.start()
    model_processes.append(p)

    # Start the remaining model inference workers without GPU
    for _ in range(1, args.num_inference_workers):
        p = multiprocessing.Process(target=model_inference_worker, args=(preprocess_queue, postprocess_queue, args, False))
        p.start()
        model_processes.append(p)

    # Start post-processing workers
    postprocess_processes = []
    for _ in range(args.num_postprocess_workers):
        p = multiprocessing.Process(target=postprocessing_worker, args=(postprocess_queue, args))
        p.start()
        postprocess_processes.append(p)

    # Enqueue files to be processed
    for index, row in chunk.iterrows():
        file_queue.put((index, row))

    # Signal the end of the file queue
    for _ in range(args.num_preprocess_workers):
        file_queue.put(None)

    # Wait for all tasks to be completed
    file_queue.join()
    preprocess_queue.join()
    postprocess_queue.join()

    # Signal the end of the preprocess queue
    for _ in range(args.num_inference_workers):
        preprocess_queue.put(None)
    # Signal the end of the postprocess queue
    for _ in range(args.num_postprocess_workers):
        postprocess_queue.put(None)

    # Terminate the worker processes
    for p in preprocess_processes:
        p.join()

    for p in model_processes:
        p.join()

    for p in postprocess_processes:
        p.join()


def main():
    parser = argparse.ArgumentParser(
        description='Process audio files using a pipeline with preprocessing, model inference, and post-processing stages.')
    parser.add_argument('--info', required=True, type=str, help="CSV file with 'input_fp' and 'output_fp' columns.")
    parser.add_argument('--model_dir', required=True, type=str, help='Directory containing the model checkpoint and meta files.')
    parser.add_argument('--input_channels', default=0, type=int, help='Number of input channels.')
    parser.add_argument('--output_channels', default=0, type=int, help='Number of output channels.')
    parser.add_argument('--input_tensor', default='input_audio/receiver_audio:0', type=str, help='Input tensor name.')
    parser.add_argument('--output_tensor', default='denoised_waveforms:0', type=str, help='Output tensor name.')
    parser.add_argument('--write_outputs_separately', default=True, type=lambda x: bool(strtobool(x)), help='Whether to write outputs separately.')
    parser.add_argument('--window_type', default='rectangular', type=str, help='Window type for stitching.')
    parser.add_argument('--block_size_in_seconds', default=10.0, type=float, help='Block size in seconds.')
    parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate.')
    parser.add_argument('--permutation_invariant', default=False, type=lambda x: bool(strtobool(x)), help='Use permutation invariant stitching.')
    parser.add_argument('--scale_input', default=False, type=lambda x: bool(strtobool(x)), help='Scale input audio.')
    parser.add_argument('--checkpoint', default=None, type=str, help='Path to model checkpoint.')
    parser.add_argument('--num_preprocess_workers', default=4, type=int, help='Number of preprocessing worker processes.')
    parser.add_argument('--num_inference_workers', default=1, type=int, help='Number of inference worker processes.')
    parser.add_argument('--num_postprocess_workers', default=4, type=int, help='Number of post-processing worker processes.')
    parser.add_argument('--chunk_size', default=100, type=int, help='Number of rows to process in each chunk.')
    parser.add_argument('--queue_size', default=50, type=int, help='Maximum size of the inter-process queues.')
    args = parser.parse_args()

    args.block_size_in_samples = 2 * int(
        round(args.block_size_in_seconds * float(args.sample_rate) / 2.0))

    info_df = pd.read_csv(args.info)

    # Initialize tqdm for tracking progress
    with tqdm(total=len(info_df), desc="Processing", unit="file") as pbar:
        start_time = time.time()

        # Process the dataframe in chunks
        for start in range(0, len(info_df), args.chunk_size):
            chunk = info_df.iloc[start:start + args.chunk_size]
            process_chunk(chunk, args)

            # Update progress
            pbar.update(len(chunk))
            elapsed_time = time.time() - start_time
            avg_time_per_file = elapsed_time / pbar.n if pbar.n > 0 else 0
            eta = avg_time_per_file * (len(info_df) - pbar.n)
            pbar.set_postfix(eta=f"{eta:.2f}s")

    print("Processing complete.")


if __name__ == "__main__":
    main()
