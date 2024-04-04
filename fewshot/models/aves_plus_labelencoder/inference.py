import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import os
from tqdm import tqdm

from fewshot.data.data import get_inference_dataloader, load_audio

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def process_dcase(audio, annotations, args):
    # TODO: UNK is not handled carefully
    print("UNK ASSUMED POS!!")
    
    annot_pos = annotations[annotations["Q"] == "POS"].reset_index()
    annot_pos_support = annot_pos.iloc[:5] # grab first five pos examples
    annot_pos_support_np = np.zeros_like(audio)
    pos_support_end = annot_pos_support["Endtime"].max()
    for i, row in annot_pos_support.iterrows():
        start_sample = int(row['Starttime'] *args.sr)
        end_sample = int(row['Endtime'] *args.sr)
        annot_pos_support_np[start_sample:end_sample] = 2
        
    annot_unk = annotations[annotations["Q"] == "UNK"].reset_index()
    annot_unk_support = annot_unk[annot_unk["Starttime"] <= pos_support_end]
    for i, row in annot_unk_support.iterrows():
        start_sample = int(row['Starttime'] *args.sr)
        end_sample = int(row['Endtime'] *args.sr)
        annot_pos_support_np[start_sample:end_sample] = 2
    
    support_end_time = annot_pos_support['Endtime'].max()+0.5
    support_end_sample = int(support_end_time*args.sr)
    support_dur_samples = int(args.sr*args.support_dur_sec)
    
    if support_end_sample <= support_dur_samples:
        support_audio = audio[max(0,support_end_sample-support_dur_samples):support_end_sample]
        support_annotations = annot_pos_support_np[max(0,support_end_sample-support_dur_samples):support_end_sample]
        # if too short, loop to support_dur_sec
        looped_audio = np.tile(support_audio, (support_dur_samples // len(support_audio) + 2,))
        looped_annotations = np.tile(support_annotations, (support_dur_samples // len(support_audio) + 2,))
        support_audio = looped_audio[:support_dur_samples]
        support_annotations = looped_annotations[:support_dur_samples]
    
    else:
        rng = np.random.default_rng(args.seed)
        # if too long, sample chunks from within the support clip. anchors are start points of chunks
        n_support_anchors = 12
        support_chunk_dur = args.support_dur_sec/n_support_anchors
        pos_starts = annot_pos_support['Starttime'].values[:n_support_anchors]
        support_anchor_starts = np.maximum(pos_starts-support_chunk_dur/2,np.zeros_like(pos_starts))
        addl_starts = rng.uniform(0,support_end_time-support_chunk_dur,n_support_anchors-len(pos_starts))
        support_anchor_starts=np.concatenate([support_anchor_starts, addl_starts])
        support_anchor_ends=np.maximum(support_anchor_starts+support_chunk_dur,support_end_time)
        support_audio = [audio[int(support_anchor_starts[i]*args.sr):int(support_anchor_ends[i]*args.sr)] for i in range(n_support_anchors)]
        support_annotations = [annot_pos_support_np[int(support_anchor_starts[i]*args.sr):int(support_anchor_ends[i]*args.sr)] for i in range(n_support_anchors)]
        shuffle=rng.permutation(np.arange(n_support_anchors))
        support_audio=[support_audio[i] for i in shuffle]
        support_annotations=[support_annotations[i] for i in shuffle]
        support_audio = np.concatenate(support_audio)[:support_dur_samples]
        support_annotations = np.concatenate(support_annotations)[:support_dur_samples]
        if len(support_audio)<support_dur_samples:
            # if the vocs were too close to the end we need to pad out some
            support_audio=np.concatenate([audio[:support_dur_samples-len(support_audio)],support_audio])
            support_annotations=np.concatenate([annot_pos_support_np[:support_dur_samples-len(support_annotations)],support_annotations])
        
        # check files:
        # import soundfile as sf
        # from matplotlib import pyplot as plt
        # sf.write("/home/jupyter/example_support.wav", support_audio, args.sr)
        # plt.plot(support_annotations)
        # plt.savefig("/home/jupyter/example_support_annot.png")
        # plt.close()
        
        # old version:
        # support_audio = audio[support_end_sample-support_dur_samples:support_end_sample]
        # support_annotations = annot_pos_support_np[support_end_sample-support_dur_samples:support_end_sample]
                
    return support_audio, support_annotations, audio

def fill_holes(m, max_hole):
    stops = m[:-1] * ~m[1:]
    stops = np.where(stops)[0]
    
    for stop in stops:
        look_forward = m[stop+1:stop+max_hole]
        if np.any(look_forward):
            next_start = np.amin(np.where(look_forward)[0]) + stop + 1
            m[stop : next_start] = True
            
    return m

def delete_short(m, min_pos):
    starts = m[1:] * ~m[:-1]

    starts = np.where(starts)[0] + 1

    clips = []

    for start in starts:
        look_forward = m[start:]
        ends = np.where(~look_forward)[0]
        if len(ends)>0:
            clips.append((start, start+np.amin(ends)))
            
    m = np.zeros_like(m).astype(bool)
    for clip in clips:
        if clip[1] - clip[0] >= min_pos:
            m[clip[0]:clip[1]] = True
        
    return m
    

def postprocess(all_query_predictions, audio_fp, args):
    pred_sr = args.sr // args.scale_factor
    audio_fn = os.path.basename(audio_fp)
    
    # convert logits -> binary
    all_query_predictions_binary = all_query_predictions >= 0
    
    # fill gaps and omit extremely short predictions
    preds = fill_holes(all_query_predictions_binary, int(pred_sr*.1))
    preds = delete_short(preds, int(pred_sr*.03))
    
    starts = preds[1:] * ~preds[:-1]
    starts = np.where(starts)[0] + 1
    
    d = {"Audiofilename" : [], "Starttime" : [], "Endtime" : []}
    
    for start in starts:
        look_forward = preds[start:]
        ends = np.where(~look_forward)[0]
        if len(ends)>0:
            end = start+np.amin(ends)
            d["Audiofilename"].append(audio_fn)
            d["Starttime"].append(start/pred_sr)
            d["Endtime"].append(end/pred_sr)
            
    d = pd.DataFrame(d)
    
    return d  

def inference_dcase(model, args, audio_fp, annotations_fp):
    model = model.to(device)
    model.eval()
    
    audio = load_audio(audio_fp, args.sr)
    annotations = pd.read_csv(annotations_fp)

    support_audio, support_annotations, query_audio = process_dcase(audio, annotations, args)
    
    query_dur_samples = int(args.query_dur_sec * args.sr)
    pad = (query_dur_samples - (query_audio.size(0) % query_dur_samples)) % query_dur_samples
    padded_query_audio = F.pad(query_audio, (0,pad))
    
    inference_dataloader = get_inference_dataloader(support_audio, support_annotations, padded_query_audio, args)
    
    all_query_predictions = []
    with torch.no_grad():
        for i, data_item in tqdm(enumerate(inference_dataloader)):
            sa, sl, qa = data_item
            query_predictions, _ = model(sa.to(device), sl.to(device), qa.to(device))
            all_query_predictions.append(query_predictions)
        
    all_query_predictions = torch.cat(all_query_predictions, dim=0)
    all_query_predictions = torch.reshape(all_query_predictions, (-1,)).cpu().numpy()
    
    all_query_predictions = all_query_predictions[:(len(query_audio) // args.scale_factor)] # omit predictions for padded region at end of query
    all_query_predictions = postprocess(all_query_predictions, audio_fp, args)
    
    return all_query_predictions
    