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
    # TODO: How to handle extremely long support?
    print("UNK ASSUMED POS!!")
    
    rng = np.random.default_rng(0)
    
    annot_pos = annotations[annotations["Q"] == "POS"].reset_index()
    annot_pos_support = annot_pos.iloc[:5] # grab first five pos examples
    annot_pos_support_tensor = torch.zeros_like(audio)
    pos_support_end = annot_pos_support["Endtime"].max()
    for i, row in annot_pos_support.iterrows():
        start_sample = int(row['Starttime'] *args.sr)
        end_sample = int(row['Endtime'] *args.sr)
        annot_pos_support_tensor[start_sample:end_sample] = 2
        
    annot_unk = annotations[annotations["Q"] == "UNK"].reset_index()
    annot_unk_support = annot_unk[annot_unk["Starttime"] <= pos_support_end]
    for i, row in annot_unk_support.iterrows():
        start_sample = int(row['Starttime'] *args.sr)
        end_sample = int(row['Endtime'] *args.sr)
        annot_pos_support_tensor[start_sample:end_sample] = 2
    
    support_end_time = annot_pos_support['Endtime'].max()+0.1
    support_end_sample = int(support_end_time*args.sr)
    
    support_audio = audio[:support_end_sample]
    support_annotations = annot_pos_support_tensor[:support_end_sample]
    
    # Sub-select from support audio so we don't end up with hours
    chunk_size_sec = 30
    chunk_size_samples = int(args.sr * chunk_size_sec)
    max_n_chunks_to_keep = 10
    
    chunks_to_keep = []
    chunks_to_maybe_keep = []
    
    for chunk_start in np.arange(0, support_end_sample, chunk_size_samples):
        chunk_end = min(chunk_start+chunk_size_samples, support_end_sample)
        annot_pos_start_sub = annot_pos[(annot_pos['Starttime'] >= chunk_start) & (annot_pos['Starttime'] < chunk_end)]
        annot_pos_end_sub = annot_pos[(annot_pos['Endtime'] >= chunk_start) & (annot_pos['Endtime'] < chunk_end)]
        if len(annot_pos_start_sub) + len(annot_pos_end_sub)>0:
            chunks_to_keep.append(chunk_start)
        else:
            chunks_to_maybe_keep.append(chunk_start)
            
    while (len(chunks_to_keep) < max_n_chunks_to_keep) and (len(chunks_to_maybe_keep) > 0):
        # choose chunks to include
        chunks_to_maybe_keep = list(rng.permutation(chunks_to_maybe_keep))
        to_add = chunks_to_maybe_keep.pop()
        chunks_to_keep.append(to_add)
        
    chunks_to_keep = sorted(chunks_to_keep)
    support_audio_new = []
    support_annot_new = []
    for chunk_start in chunks_to_keep:
        chunk_end = min(chunk_start+chunk_size_samples, support_end_sample)
        support_audio_new.append(support_audio[chunk_start:chunk_end])
        support_annot_new.append(support_annotations[chunk_start:chunk_end])
        
    support_audio = torch.cat(support_audio_new)
    support_annotations = torch.cat(support_annot_new)
    
    assert len(support_audio) == len(support_annotations)
    
    print(f"Support audio, before subsampling: {support_end_time}. After subsampling: {len(support_audio)/args.sr}")
    
    return support_audio, support_annotations, audio

def fill_holes(m, max_hole):
    stops = m[:-1] * ~m[1:]
    stops = np.where(stops)[0]
    
    for stop in stops:
        look_forward = m[stop+1:stop+1+max_hole]
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
    
    all_query_predictions_binary = all_query_predictions >= 0 # TODO
    
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
    print(f"Inference for {audio_fp}")
    
    fn = os.path.basename(audio_fp)
    
    # loading for speedup
    np_fp = os.path.join(args.experiment_dir, fn[:-4]+".npy")
    if False: #os.path.exists(np_fp):
        all_query_predictions = np.load(np_fp)
    #
        
    else:
        model = model.to(device)
        model.eval()

        audio = load_audio(audio_fp, args.sr)
        annotations = pd.read_csv(annotations_fp)

        support_audio, support_annotations, query_audio = process_dcase(audio, annotations, args)
        
        # pad etc
        
        ## loop audio at offset to provide a windowing effect
        support_training_dur_samples = int(args.sr*args.support_dur_sec)
        support_dur_samples = support_audio.size(0)
        
        assert model.audio_chunk_size_samples % 2 == 0
        assert support_dur_samples >= model.audio_chunk_size_samples
        
        remainder = support_dur_samples % model.audio_chunk_size_samples
        if remainder >= model.audio_chunk_size_samples//2:
            to_cut = remainder-model.audio_chunk_size_samples//2
        else:
            to_cut = remainder+model.audio_chunk_size_samples//2
                    
        halfwindowed_audio_len = support_audio[to_cut:].size(0)
        assert halfwindowed_audio_len % model.audio_chunk_size_samples == model.audio_chunk_size_samples//2, "incorrect windowing math!"
        support_audio = torch.cat((support_audio[to_cut:], support_audio))
        support_annotations = torch.cat((support_annotations[to_cut:], support_annotations))
        
        # Pad out so we don't have empty sounds
        support_pad = (model.audio_chunk_size_samples - (support_audio.size(0) % model.audio_chunk_size_samples)) % model.audio_chunk_size_samples
        if support_pad>0:
            support_audio = torch.cat((support_audio, support_audio[:support_pad]))
            support_annotations = torch.cat((support_annotations, support_annotations[:support_pad]))
        
        # Pad query to match training length
        query_dur_samples = int(args.query_dur_sec * args.sr)
        query_pad = (query_dur_samples - (query_audio.size(0) % query_dur_samples)) % query_dur_samples
        if query_pad>0:
            query_audio = F.pad(query_audio, (0,query_pad))
       
        assert len(support_annotations) == len(support_audio)
        inference_dataloader = get_inference_dataloader(support_audio, support_annotations, query_audio, args)

        all_query_predictions = []
        with torch.no_grad():
            for i, data_item in tqdm(enumerate(inference_dataloader)):
                sa, sl, qa = data_item
                query_predictions, _ = model(sa.to(device), sl.to(device), qa.to(device), temperature=1) # lower temperature gives more weight to high confidence votes; use as hyperparam for long support sets?
                all_query_predictions.append(query_predictions)
        
        all_query_predictions = torch.cat(all_query_predictions, dim=0)
        assert all_query_predictions.size(1) % 4 == 0
        quarter_window = all_query_predictions.size(1)//4
        
        all_query_predictions_windowed = torch.reshape(all_query_predictions[:, quarter_window:-quarter_window], (-1,))
        all_query_predictions = torch.cat((all_query_predictions[0,:quarter_window], all_query_predictions_windowed, all_query_predictions[-1,-quarter_window:])).cpu().numpy()
        
        #remove query padding        
        if query_pad // args.scale_factor > 0:
            all_query_predictions = all_query_predictions[:-(query_pad // args.scale_factor)] # omit predictions for padded region at end of query

        # save np array of predictions
        fn = os.path.basename(audio_fp)
        np.save(os.path.join(args.experiment_dir, fn[:-4]+".npy"), all_query_predictions)
    
    d = postprocess(all_query_predictions, audio_fp, args)
    
    # Save raven st
    st = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : []}
    for i, row in d.iterrows():
        st["Begin Time (s)"].append(row["Starttime"])
        st["End Time (s)"].append(row["Endtime"])
        st["Annotation"].append("POS")
    st = pd.DataFrame(st)
    st.to_csv(os.path.join(args.experiment_dir, fn[:-4]+'.txt'), index=False, sep='\t')
    
    return d
    