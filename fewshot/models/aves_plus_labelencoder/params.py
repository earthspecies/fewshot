import argparse
import torch
import numpy as np
import random
import logging
import os
import yaml

def parse_args(args,allow_unknown=False):
    parser = argparse.ArgumentParser()
    
    # General 
    parser.add_argument('--name', type = str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project-dir', type=str, default="/home/jupyter/fewshot/projects/aves_plus_labelencoder_med")
    parser.add_argument('--previous-checkpoint-fp', type=str, default=None, help="path to checkpoint of previously trained detection model")
    
    # Model
    parser.add_argument('--aves-config-fp', type=str, default = "/home/jupyter/fewshot/weights/aves-base-bio.torchaudio.model_config.json")
    parser.add_argument('--aves-url', type=str, default = "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.pt")
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--scale-factor', type=int, default = 320, help = "downscaling performed by aves")
    parser.add_argument('--audio-chunk-size-sec', type=float, default=4, help="chunk audio before passing into aves encoder")
    parser.add_argument('--embedding-dim', type=int, default=768, help="dim of audio embedding computed by aves")
    parser.add_argument('--label-encoder-dim', type=int, default=512, help="dim of embeddings computed by label encoder attention layers. Default follows BERT Small")
    parser.add_argument('--label-encoder-depth', type=int, default=4, help="n of mhsa layers in transformer label encoder. Default follows BERT Small")
    parser.add_argument('--label-encoder-heads', type=int, default=8, help="n heads in each mhsa layer. note dim per head is dim/n heads. Default follows BERT Small")    
    parser.add_argument('--support-dur-sec', type=float, default=24, help="dur of support audio fed into model")
    parser.add_argument('--query-dur-sec', type=float, default=4, help="dur of query audio fed into model")
    parser.add_argument('--atst-frame', action="store_true", help="Skip the final transformer and label encoder")
    parser.add_argument('--atst-model-path', type=str, default="/home/jupyter/fewshot/weights/atstframe_base.ckpt")
    
    # Training
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--lr', type=float, default=.0001) 
    parser.add_argument('--unfreeze-encoder-step', type=int, default=5000)
    parser.add_argument('--checkpoint-frequency', type=int, default=200)
    parser.add_argument('--n-train-steps', type=int, required=True)
    parser.add_argument('--clip-grad-norm', type=float, default=3.0)
    parser.add_argument('--n-steps-warmup', type=int, default=10000)
    parser.add_argument('--log-steps', type=int, default=100)
    parser.add_argument('--wandb', action="store_true", help="log to wandb")
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help="accumulate gradients over this many steps")
    parser.add_argument('--window-train-support', action="store_true", help="window the support audio during training")
    parser.add_argument('--mixed-precision', action="store_true", help="use mixed precision training")
    
    # Data
    parser.add_argument('--TUT-background-audio-info-fp', type = str, default='/home/jupyter/data/fewshot_data/data_medium/TUT_background_audio_info.csv')
    parser.add_argument('--audioset-background-audio-info-fp', type = str, default='/home/jupyter/data/fewshot_data/data_medium/audioset_background_audio_info.csv')
    parser.add_argument('--xeno-canto-background-audio-info-fp', type = str, default='/home/jupyter/data/fewshot_data/data_medium/xeno_canto_background_audio_info.csv')
    parser.add_argument('--pseudovox-info-fp', type=str, default='/home/jupyter/data/fewshot_data/data_medium/pseudovox_bio.csv')
    parser.add_argument('--nonbio-pseudovox-info-fp', type=str, default='/home/jupyter/data/fewshot_data/data_medium/pseudovox_nonbio.csv')
    parser.add_argument('--scenarios', type=str, default="normal,disjunction_within_species,generalization_within_species,low_snr,fine_grained_snr,fine_grained_general", help = "csv of scenarios to choose from for constructing examples")
    
    parser.add_argument('--min-background-duration', type=float, default = 6, help = "the min dur in seconds that a file is allowed to be, in order for it to be used as background audio.")
    parser.add_argument('--max-pseudovox-duration', type=float, default=12, help= "the max dur in seconds that a pseudovox may be")
    parser.add_argument('--min-cluster-size', type = int, default=4, help="the minimum number of pseudovox in a cluster, in order for that cluster to be included as an option")
    parser.add_argument('--min-cluster-size-for-longish-pseudovox', type = int, default=2, help = "the min cluster size when a pseudovox is >=1 sec long, we allow this because there aren't that many of them")
    parser.add_argument('--nonbio-min-cluster-size', type = int, default=4, help="the minimum number of nonbio pseudovox in a cluster, in order for that cluster to be included as an option")
    parser.add_argument('--birdnet-confidence-strict-lower-bound', type=float, default=0, help="will filter out examples with birdnet confidence <= this value. Mostly used to remove pseudovox with no sounds of interest")    
    
    # Evaluation
    parser.add_argument('--dcase-ref-files-path', default="/home/jupyter/fewshot/data/DCASE2024_Development_Set/Validation_Set/", type=str, help="Path to parent dir of DCASE files to evaluate on")
    parser.add_argument('--dcase-evaluation-manifest-fp', default="/home/jupyter/fewshot/data/DCASE2024_Development_Set/Validation_Set_manifest.csv", type=str, help="Path to manifest of DCASE files to evaluate on")
    # parser.add_argument('--inference-temperature', default = 1, type=float, help="Deprecated")
    parser.add_argument('--inference-threshold', default = None, type=float, help = "prob threshold to count as positive; None uses adaptive threshold based on duration of support events")
    parser.add_argument('--inference-n-chunks-to-keep', default=10, type=int, help="longer means support during inference will be longer")
    parser.add_argument('--inference-chunk-size-sec', default =16, type=float, help ="duration of audio chunks included in support")
    parser.add_argument('--window-inference-support', action="store_true", help="window the support audio during inference")
    parser.add_argument('--window-inference-query', action="store_true", help="window the query audio during inference")
    parser.add_argument('--inference-hard-negative-sampling', action="store_true", help="sample hard negatives for prompt")
    
    args = parser.parse_args(args)
    
    setattr(args, "n_synthetic_examples", args.batch_size * args.n_train_steps)

    check_config(args)

    return args

def save_params(args):
    """ Save a copy of the params used for this experiment """
    logging.info("Params:")
    params_file = os.path.join(args.experiment_dir, "params.yaml")
    args_dict = {}
    for name in sorted(vars(args)):
        val = getattr(args, name)
        logging.info(f"  {name}: {val}")
        args_dict[name] = val
    with open(params_file, "w") as f:
        yaml.dump(args_dict, f)
        
def load_params(fp):
    with open(fp, 'r') as f:
        args_dict = yaml.safe_load(f)

    args = argparse.Namespace()

    for key in args_dict:
        setattr(args, key, args_dict[key])

    return args

def check_config(args):
    assert args.support_dur_sec % args.audio_chunk_size_sec == 0
    # assert args.query_dur_sec % args.audio_chunk_size_sec == 0
    assert args.audio_chunk_size_sec * args.sr % args.scale_factor == 0
    if args.atst_frame:
        assert args.embedding_dim == 9216
        assert args.scale_factor == 640
        assert args.audio_chunk_size_sec == 10
