import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import os
from tqdm import tqdm
from einops import rearrange
import argparse

from fewshot.data.data import get_inference_dataloader, load_audio
from fewshot.models.audio_plus_labelencoder.params import load_params
from fewshot.models.audio_plus_labelencoder.model import FewShotModel
from fewshot.models.audio_plus_labelencoder.inference import cache_support_encoded, forward_cached, get_probs, get_threshold, fill_holes, delete_short

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def subselect_support(support_audio, support_annotations, support_annotations_st, args):
    # Sub-select from support audio so we don't end up with hours
        
    rng = np.random.default_rng(0)
    
    chunk_size_samples = int(args.sr * args.inference_chunk_size_sec)
    
    chunks_to_keep = []
    chunks_to_maybe_keep = []
    chunks_to_maybe_keep_probs = []
    
    support_end_sample= support_audio.size(0)+1  
    
    for chunk_start in np.arange(0, support_end_sample, chunk_size_samples):
        chunk_end = min(chunk_start+chunk_size_samples, support_end_sample)
        annot_pos_start_sub = support_annotations_st[(support_annotations_st['Begin Time (s)'] >= chunk_start/args.sr) & (support_annotations_st['Begin Time (s)'] < chunk_end/args.sr)]
        annot_pos_end_sub = support_annotations_st[(support_annotations_st['End Time (s)'] >= chunk_start/args.sr) & (support_annotations_st['End Time (s)'] < chunk_end/args.sr)]
        annot_pos_long_sub = support_annotations_st[(support_annotations_st['Begin Time (s)'] < chunk_start/args.sr) &(support_annotations_st['End Time (s)'] >= chunk_end/args.sr)]
        if len(annot_pos_start_sub) + len(annot_pos_end_sub) + len(annot_pos_long_sub) >0:
            chunks_to_keep.append(chunk_start)
        else:
            chunks_to_maybe_keep.append(chunk_start)
    
    while (len(chunks_to_keep) < args.inference_n_chunks_to_keep) and (len(chunks_to_maybe_keep) > 0):
        # choose chunks to include
        chunks_to_maybe_keep = list(rng.permutation(chunks_to_maybe_keep))
        to_add = chunks_to_maybe_keep.pop()
        chunks_to_keep.append(to_add)
            
    if args.support_duration_limit_sec is not None:
        limit_n_chunks = args.support_duration_limit_sec // args.inference_chunk_size_sec
        if len(chunks_to_keep) > limit_n_chunks:
            chunks_to_keep = rng.permutation(chunks_to_keep)[:limit_n_chunks]
        
    chunks_to_keep = sorted(chunks_to_keep)
    support_audio_new = []
    support_annot_new = []
    for chunk_start in chunks_to_keep:
        chunk_end = min(chunk_start+chunk_size_samples, support_end_sample)
        support_audio_new.append(support_audio[chunk_start:chunk_end])
        support_annot_new.append(support_annotations[chunk_start:chunk_end])
        
    support_audio = torch.cat(support_audio_new)
    support_annotations = torch.cat(support_annot_new)
    
    return support_audio, support_annotations

def postprocess(all_query_predictions, args, min_vox_dur_support, vox_durs_support):
    pred_sr = args.sr // args.scale_factor
    
    threshold = get_threshold(all_query_predictions, args, min_vox_dur_support, vox_durs_support)
    
    all_query_predictions_binary = all_query_predictions >= threshold

    # fill gaps and omit extremely short predictions
    max_hole_size_sec = np.clip(0.5*min_vox_dur_support, 0, 1)
    min_vox_dur_sec = min(0.5, 0.5*min_vox_dur_support)

    preds = fill_holes(all_query_predictions_binary, int(pred_sr*max_hole_size_sec))
    preds = delete_short(preds, int(pred_sr*min_vox_dur_sec))

    starts = preds[1:] * ~preds[:-1]
    starts = np.where(starts)[0] + 1

    if preds[0]:
        starts = np.concatenate(np.zeros(1,), starts)

    d = {"Annotation" : [], "Begin Time (s)" : [], "End Time (s)" : []}

    for start in starts:
        look_forward = preds[start:]
        ends = np.where(~look_forward)[0]
        if len(ends)>0:
            end = start+np.amin(ends)
        else:
            end = len(preds)-1
        d["Annotation"].append(args.focal_annotation_label)
        d["Begin Time (s)"].append(start/pred_sr)
        d["End Time (s)"].append(end/pred_sr)

    d = pd.DataFrame(d)
       
    return d

def infer(model_path,
          atst_model_path,
          support_audio_fp, 
          support_selection_table_fp, 
          query_audio_fp, 
          focal_annotation_label="POS",
          window_inference_query=True,
          window_inference_support=True,
          inference_normalize_rms=0.005,
          inference_threshold=None,
          inference_n_chunks_to_keep=5,
          inference_chunk_size_sec=8,
          support_duration_limit_sec=None
         ):
    """
    model_path (str) : path to model checkpoint
    atst_path (str) : path to atst checkpoint
    model_args_fp (str) : path to .yaml model parameters
    support_audio_fp (str) : path to support audio
    support_selection_table_fp (str) : path to Raven-style tsv for selection table
    query_audio_fp (str) : path to audio that model will make predictions for
    focal_annotation_label (str) : label of class in annotation column that should be detected
    inference_normalize_rms=0.005 (float or None) : normalize so that support audio has this rms. Results are sensitive to this number, future versions will try to remove
    window_inference_query=True (bool) : whether to window query during inference, recommended True
    window_inference_support=True (bool) : whether to window support during inference, recommended True
    inference_threshold=None (float or None) : threshold over which detections are considered as positive
    inference_chunk_size_sec=8 (float) : divide support audio into chunks of this size, just for the purpose of sub-sampling
    inference_n_chunks_to_keep=5 (int) : how many of these chunks should we try to keep?
    support_duration_limit_sec=None (float or None) : if float, will apply a strict limit to the duration of the support audio. Recommended for extremely long & dense support, to limit runtime.    
    """
    
    args = argparse.Namespace()
    args.atst_model_path = atst_model_path
    args.focal_annotation_label = focal_annotation_label
    args.inference_normalize_rms = inference_normalize_rms
    args.window_inference_query=window_inference_query
    args.window_inference_support=window_inference_support
    args.inference_threshold = inference_threshold
    args.inference_chunk_size_sec = inference_chunk_size_sec
    args.inference_n_chunks_to_keep = inference_n_chunks_to_keep
    args.support_duration_limit_sec = support_duration_limit_sec
    
    ## hard-coded args:
    args.sr = 16000
    args.support_dur_sec=40
    args.query_dur_sec=10
    args.window_train_support=True
    args.scale_factor=640
    args.embedding_dim=9216
    args.label_encoder_dim=512
    args.label_encoder_depth=4
    args.label_encoder_heads=8
    args.audio_chunk_size_sec=10
    
    print("Init model")
    model=FewShotModel(args)
    
    print(f"loading model weights from {model_path}")
    cp = torch.load(model_path)
    model.load_state_dict(cp["model_state_dict"])
    
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    support_audio = load_audio(support_audio_fp, args.sr)
    query_audio = load_audio(query_audio_fp, args.sr)
    support_st = pd.read_csv(support_selection_table_fp, sep='\t')
    support_st = support_st[support_st['Annotation'] == args.focal_annotation_label]
    
    support_annotations = torch.zeros_like(support_audio)
    for i, row in support_st.iterrows():
        begin_sample = int(row['Begin Time (s)'] * args.sr)
        end_sample = int(row["End Time (s)"] * args.sr)
        support_annotations[begin_sample:end_sample] = 2
    
    support_audio, support_annotations = subselect_support(support_audio, support_annotations, support_st, args)
    
    min_vox_dur_support = (support_st["End Time (s)"] - support_st["Begin Time (s)"]).min()
    vox_durs_support = list(support_st["End Time (s)"] - support_st["Begin Time (s)"])
    
    all_query_predictions = get_probs(model, support_audio, support_annotations, query_audio, args)
    st = postprocess(all_query_predictions, args, min_vox_dur_support, vox_durs_support)
    
    return st
    
