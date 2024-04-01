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
    parser.add_argument('--project-dir', type=str, default="/home/jupyter/fewshot/projects/aves_plus_labelencoder_small")
    parser.add_argument('--previous-checkpoint-fp', type=str, default=None, help="path to checkpoint of previously trained detection model")
    
    # Model
    parser.add_argument('--aves-config-fp', type=str, default = "/home/jupyter/fewshot/weights/aves-base-bio.torchaudio.model_config.json")
    parser.add_argument('--aves-url', type=str, default = "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.pt")
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--scale-factor', type=int, default = 320, help = "downscaling performed by aves")
    parser.add_argument('--audio-chunk-size-sec', type=float, default=6, help="chunk audio before passing into aves encoder")
    parser.add_argument('--embedding-dim', type=int, default=768, help="dim of audio embedding computed by aves")
    parser.add_argument('--label-encoder-dim', type=int, default=512, help="dim of embeddings computed by label encoder attention layers")
    parser.add_argument('--label-encoder-depth', type=int, default=3, help="n of mhsa layers in transformer label encoder")
    parser.add_argument('--label-encoder-heads', type=int, default=16, help="n heads in each mhsa layer. note dim per head is dim/n heads")    
    parser.add_argument('--support-dur-sec', type=float, default=30, help="dur of support audio fed into model")
    parser.add_argument('--query-dur-sec', type=float, default=6, help="dur of query audio fed into model")
    
    # Training
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.0001) 
    parser.add_argument('--unfreeze-encoder-step', type=int, default=10000)
    parser.add_argument('--checkpoint-frequency', type=int, default=200)
    parser.add_argument('--n-train-steps', type=int, required=True)
    
    # Data
    parser.add_argument('--background-audio-dir', type = str, default = '/home/jupyter/fewshot/data/data_small/audio_trimmed', help="path to dir with background audio")
    parser.add_argument('--pseudovox-audio-dir', type = str, default = '/home/jupyter/fewshot/data/data_small/pseudovox', help="path to dir with pseudovox audio")
    
    parser.add_argument('--background-audio-info-fp', type = str, default='/home/jupyter/fewshot/data/data_small/background_audio_info.csv')
    parser.add_argument('--pseudovox-info-fp', type=str, default='/home/jupyter/fewshot/data/data_small/pseudovox_manifest_birdnet_filtered.csv')
    
    parser.add_argument('--min-background-duration', type=float, default = 15, help = "the min dur in seconds that a file is allowed to be, in order for it to be used as background audio.")
    parser.add_argument('--max-pseudovox-duration', type=float, default=5, help= "the max dur in seconds that a pseudovox may be")
    parser.add_argument('--min-cluster-size', type = int, default=10, help="the minimum number of pseudovox in a cluster, in order for that cluster to be included as an option")
    parser.add_argument('--birdnet-confidence-strict-lower-bound', type=float, default=0, help="will filter out examples with birdnet confidence <= this value. Mostly used to remove pseudovox with no sounds of interest")
    parser.add_argument('--cluster-column', type=str, default='birdnet_prediction', choices=['prediction', 'birdnet_prediction'], help="name of column in manifest to use for forming groups of calls")
    parser.add_argument('--return-timestamps', type=bool, default=False, help="True to return timestamps from the dataloader")
    
    
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
    assert args.query_dur_sec % args.audio_chunk_size_sec == 0
    assert args.audio_chunk_size_sec * args.sr % args.scale_factor == 0