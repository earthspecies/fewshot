import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import os
from tqdm import tqdm
from einops import rearrange

from fewshot.data.data import get_inference_dataloader, load_audio, apply_windowing

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def process_dcase(audio, annotations, sr, chunk_size_sec, max_n_chunks_to_keep, scale_factor=None, support_probs = None):
    # TODO: UNK is not handled carefully
    # TODO: max_n_chunks_to_keep is over-ridden when there are more chunks that contain POS events than max_n_chunks_to_keep
    # print("UNK ASSUMED POS!!")
    
    if support_probs is None:
        hard_negative_sampling = False
    else:
        hard_negative_sampling = True
    
    rng = np.random.default_rng(0)
        
    annotations = annotations.sort_values(by="Starttime").reset_index()
        
    annot_pos = annotations[annotations["Q"] == "POS"].reset_index()
    annot_pos_support = annot_pos.iloc[:5] # grab first five pos examples
    annot_pos_support_tensor = torch.zeros_like(audio)
    pos_support_end = annot_pos_support["Endtime"].max()
    for i, row in annot_pos_support.iterrows():
        start_sample = int(row['Starttime'] * sr)
        end_sample = int(row['Endtime'] * sr)
        annot_pos_support_tensor[start_sample:end_sample] = 2
        
    annot_unk = annotations[annotations["Q"] == "UNK"].reset_index()
    annot_unk_support = annot_unk[annot_unk["Starttime"] <= pos_support_end]
    for i, row in annot_unk_support.iterrows():
        start_sample = int(row['Starttime'] * sr)
        end_sample = int(row['Endtime'] * sr)
        annot_pos_support_tensor[start_sample:end_sample] = 2
    
    support_end_time = annot_pos_support['Endtime'].max()+0.1
    support_end_sample = int(support_end_time* sr)
    
    support_audio = audio[:support_end_sample]
    support_annotations = annot_pos_support_tensor[:support_end_sample]
    
    # Sub-select from support audio so we don't end up with hours
    chunk_size_samples = int(sr * chunk_size_sec)
    
    chunks_to_keep = []
    chunks_to_maybe_keep = []
    chunks_to_maybe_keep_probs = []
    
    for chunk_start in np.arange(0, support_end_sample, chunk_size_samples):
        chunk_end = min(chunk_start+chunk_size_samples, support_end_sample)
        annot_pos_start_sub = annot_pos[(annot_pos['Starttime'] >= chunk_start/sr) & (annot_pos['Starttime'] < chunk_end/sr)]
        annot_pos_end_sub = annot_pos[(annot_pos['Endtime'] >= chunk_start/sr) & (annot_pos['Endtime'] < chunk_end/sr)]
        annot_pos_long_sub = annot_pos[(annot_pos['Starttime'] < chunk_start/sr) &(annot_pos['Endtime'] >= chunk_end/sr)]
        if len(annot_pos_start_sub) + len(annot_pos_end_sub) + len(annot_pos_long_sub) >0:
            chunks_to_keep.append(chunk_start)
        else:
            chunks_to_maybe_keep.append(chunk_start)
            if hard_negative_sampling:
                chunk_probs = support_probs[chunk_start//scale_factor:chunk_end//scale_factor]
                if len(chunk_probs)>0:
                    chunks_to_maybe_keep_probs.append(float(chunk_probs.max()))
                else:
                    chunks_to_maybe_keep_probs.append(0)
    
    if hard_negative_sampling:
        # This is the situation where we have previously gathered probs for the full support audio, in order to do hard negative sampling
        srtidx = np.argsort(chunks_to_maybe_keep_probs)
        chunks_to_maybe_keep = np.array(chunks_to_maybe_keep)
        chunks_to_maybe_keep = list(chunks_to_maybe_keep[srtidx]) # sort in ascending probability order
        
        while (len(chunks_to_keep) < max_n_chunks_to_keep) and (len(chunks_to_maybe_keep) > 0):
            # choose chunks to include
            to_add = chunks_to_maybe_keep.pop() # get last chunk, i.e. the one with highest prob
            chunks_to_keep.append(to_add)
    
    else:
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
    
    query_audio = audio[support_end_sample:]
            
    assert len(support_audio) == len(support_annotations)
    
    print(f"Support audio, before subsampling: {support_end_time}. After subsampling: {len(support_audio)/sr}")
    
    min_vox_dur_support = (annot_pos_support["Endtime"] - annot_pos_support["Starttime"]).min()
    vox_durs_support = list(annot_pos_support["Endtime"] - annot_pos_support["Starttime"])
    
    return support_audio, support_annotations, query_audio, support_end_time, min_vox_dur_support, vox_durs_support


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
    
def get_threshold(all_query_predictions, args, min_vox_dur_support, vox_durs_support):
    if args.inference_threshold is None:
        """
        Adaptive thresholding method.
        """
        print("Using adaptive thresholding method")
        pred_sr = args.sr // args.scale_factor
        vox_durs_mean = np.mean(vox_durs_support)
        vox_durs_std = np.std(vox_durs_support, ddof=1)

        def likelihood_of_dur(x):
            return np.exp(-0.5*((x-vox_durs_mean)/vox_durs_std)**2)

        def prior_on_threshold(t):
            return np.exp(-0.5*(t**2)) # prior: standard normal distribution on logits

        best_likelihood = 0.1
        best_threshold = 0

        for threshold in np.arange(-2,2.01,0.05):

            all_query_predictions_binary = all_query_predictions >= 1 / (1 + np.exp(-threshold))

            # fill gaps and omit extremely short predictions
            max_hole_size_sec = np.clip(0.5*min_vox_dur_support, 0, 1) #np.clip(0.5*min_vox_dur_support, 0.2, 1)
            min_vox_dur_sec = min(0.5, 0.5*min_vox_dur_support)

            preds = fill_holes(all_query_predictions_binary, int(pred_sr*max_hole_size_sec))
            preds = delete_short(preds, int(pred_sr*min_vox_dur_sec))

            starts = preds[1:] * ~preds[:-1]
            starts = np.where(starts)[0] + 1

            if preds[0]:
                starts = np.concatenate(np.zeros(1,), starts)

            d = {"Starttime" : [], "Endtime" : []}

            for start in starts:
                look_forward = preds[start:]
                ends = np.where(~look_forward)[0]
                if len(ends)>0:
                    end = start+np.amin(ends)
                else:
                    end = len(preds)-1
                d["Starttime"].append(start/pred_sr)
                d["Endtime"].append(end/pred_sr)

            d = pd.DataFrame(d)

            durations = (d["Endtime"] - d["Starttime"]).values
            if len(durations) == 0:
                likelihood = -1
            else:
                likelihood = likelihood_of_dur(durations).mean() * prior_on_threshold(threshold)
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_threshold = threshold

        threshold = 1 / (1 + np.exp(-best_threshold))
        print(f"Found best threshold {threshold}, based on distribution of durations")
        return threshold
    else:
        return args.inference_threshold


def postprocess(all_query_predictions, audio_fp, args, time_shift_sec, min_vox_dur_support, vox_durs_support):
    pred_sr = args.sr // args.scale_factor
    audio_fn = os.path.basename(audio_fp)
    
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

    d = {"Audiofilename" : [], "Starttime" : [], "Endtime" : []}

    for start in starts:
        look_forward = preds[start:]
        ends = np.where(~look_forward)[0]
        if len(ends)>0:
            end = start+np.amin(ends)
        else:
            end = len(preds)-1
        d["Audiofilename"].append(audio_fn)
        d["Starttime"].append(start/pred_sr + time_shift_sec)
        d["Endtime"].append(end/pred_sr + time_shift_sec)

    d = pd.DataFrame(d)
       
    return d

def cache_support_encoded(model, support_audio, support_labels, args):
    
    # pad and normalize audio
    support_pad = (model.audio_chunk_size_samples - (support_audio.size(1) % model.audio_chunk_size_samples)) % model.audio_chunk_size_samples
    # if support_pad>0:
    #     support_audio = torch.cat((support_audio, support_audio[:,:support_pad]), dim=1)
    
    if args.inference_normalize_rms is not None: #not args.atst_frame:
        normalization_factor = torch.std(support_audio)/args.inference_normalize_rms
        normalization_factor = torch.maximum(normalization_factor, torch.full_like(normalization_factor, 1e-6))
        support_audio = (support_audio - torch.mean(support_audio, dim=1,keepdim=True)) / normalization_factor
    
    while support_pad>0:
        support_audio = torch.cat((support_audio, support_audio[:,:support_pad]), dim=1)
        support_pad = (model.audio_chunk_size_samples - (support_audio.size(1) % model.audio_chunk_size_samples)) % model.audio_chunk_size_samples
    
    
    
    support_len_samples = support_audio.size(1)
    support_audio_subs_encoded = []
    start_samples = []
    
    for start_sample in range(0, support_len_samples, model.audio_chunk_size_samples):
        support_audio_sub = support_audio[:, start_sample:start_sample+model.audio_chunk_size_samples]
        support_audio_sub_encoded = model.audio_encoder(support_audio_sub)
        support_audio_subs_encoded.append(support_audio_sub_encoded)
        start_samples.append(start_sample)
        
    return support_audio_subs_encoded, start_samples


def forward_cached(model, support_audio, support_audio_encoded, start_samples, support_labels, query_audio, args, query_labels=None):
    """
    Input
        support_audio (Tensor): (batch, time) (at audio_sr)
        support_labels (Tensor): (batch, time) (at audio_sr)
        query_audio (Tensor): (batch, time) (at audio_sr)
        query_labels (Tensor): (batch, time) (at audio_sr)
    Output
        logits (Tensor): (batch, query_time/scale_factor) (at audio_sr / scale factor)
    """
    
    #NOTE: ATST has its own MinMax scaler
    if args.inference_normalize_rms is not None: #not args.atst_frame:
        normalization_factor = torch.std(support_audio)/args.inference_normalize_rms
        normalization_factor = torch.maximum(normalization_factor, torch.full_like(normalization_factor, 1e-6))
        query_audio = (query_audio - torch.mean(query_audio, dim=1,keepdim=True)) / normalization_factor

    # Pad support so we don't have empty sounds
    support_pad = (model.audio_chunk_size_samples - (support_audio.size(1) % model.audio_chunk_size_samples)) % model.audio_chunk_size_samples
    while support_pad>0:
        support_labels = torch.cat((support_labels, support_labels[:,:support_pad]), dim=1)
        support_pad = (model.audio_chunk_size_samples - (support_labels.size(1) % model.audio_chunk_size_samples)) % model.audio_chunk_size_samples

    # encode audio and labels
    query_logits = []
    query_confidences = []

    query_audio_encoded = model.audio_encoder(query_audio) # (batch, embedding_dim, time/scale_factor)
    
    for support_audio_sub_encoded, start_sample in zip(support_audio_encoded, start_samples):
        support_audio_sub_encoded = support_audio_sub_encoded[:query_audio_encoded.size(0),...]
        
        support_labels_sub = support_labels[:, start_sample:start_sample+model.audio_chunk_size_samples]
        support_labels_sub_downsampled = F.max_pool1d(support_labels_sub.unsqueeze(1), model.args.scale_factor, padding=0).squeeze(1) # (batch, time/scale_factor). 0=NEG 1=UNK 2=POS

        l, c = model.label_encoder(support_audio_sub_encoded, support_labels_sub_downsampled, query_audio_encoded) # each output: (batch, query_time/scale_factor). 

        c_shape = c.size() # b t c

        query_logits.append(l)
        c = torch.reshape(c, (-1, c_shape[2]))
        query_confidences.append(c)
        
    query_confidences = torch.stack(query_confidences, 1) # bt n_support c
    cls_token = model.cls_token.expand(query_confidences.size(0), -1, -1) # bt 1 c
    query_confidences = torch.cat([cls_token, query_confidences], dim=1)
    query_logits = model.confidence_transformer(query_confidences)[:,0,:].squeeze(-1).squeeze(-1) #bt 1 1 -> bt
    query_logits = torch.reshape(query_logits, (c_shape[0], c_shape[1])) # b t
    query_probs = torch.sigmoid(query_logits)

    return query_probs

def get_probs(model, support_audio, support_annotations, query_audio, args):
    
    # Pad support so we don't have empty sounds
    # support_pad = (model.audio_chunk_size_samples - (support_audio.size(0) % model.audio_chunk_size_samples)) % model.audio_chunk_size_samples
    # if support_pad>0:
    #     support_audio = torch.cat((support_audio, support_audio[:support_pad]))
    #     support_annotations = torch.cat((support_annotations, support_annotations[:support_pad]))
    
    support_pad = (model.audio_chunk_size_samples - (support_audio.size(0) % model.audio_chunk_size_samples)) % model.audio_chunk_size_samples
    while support_pad>0:
        support_audio = torch.cat((support_audio, support_audio[:support_pad]))
        support_annotations = torch.cat((support_annotations, support_annotations[:support_pad]))
        support_pad = (model.audio_chunk_size_samples - (support_audio.size(0) % model.audio_chunk_size_samples)) % model.audio_chunk_size_samples

    # pad etc
    if args.window_inference_support:
        support_audio, support_annotations = apply_windowing(support_audio, support_annotations, model.audio_chunk_size_samples)

    # Pad query to match training length
    query_dur_samples = int(args.query_dur_sec * args.sr)
    query_pad = (query_dur_samples - (query_audio.size(0) % query_dur_samples)) % query_dur_samples
    if query_pad>0:
        query_audio = F.pad(query_audio, (0,query_pad)) # TODO: add attention mask

    if args.window_inference_query:
        assert model.audio_chunk_size_samples % 2 == 0
    assert len(support_annotations) == len(support_audio)

    inference_dataloader = get_inference_dataloader(support_audio, support_annotations, query_audio, args)
    all_query_predictions = []

    with torch.no_grad():
        cached_support_encoded = None
        for i, data_item in tqdm(enumerate(inference_dataloader)):
            sa, sl, qa = data_item
            sa = sa.to(device)

            if cached_support_encoded is None:
                cached_support_encoded, start_samples = cache_support_encoded(model, sa.to(device), sl.to(device), args)

            query_predictions = forward_cached(model, sa.to(device), cached_support_encoded, start_samples, sl.to(device), qa.to(device), args)
            all_query_predictions.append(query_predictions)

    all_query_predictions = torch.cat(all_query_predictions, dim=0).cpu().numpy()

    if args.window_inference_query:
        left_quarterwindow = all_query_predictions.shape[1] // 4
        right_quarterwindow = (all_query_predictions.shape[1] // 2) - left_quarterwindow
        middle_bits = all_query_predictions[:,left_quarterwindow:-right_quarterwindow].reshape(-1)
        far_left_bit = all_query_predictions[0,:left_quarterwindow]
        far_right_bit = all_query_predictions[-1,-right_quarterwindow:]
        all_query_predictions = np.concatenate((far_left_bit, middle_bits, far_right_bit))

    else:
        all_query_predictions = all_query_predictions.reshape(-1)

    # Remove query padding if necessary
    if query_pad // args.scale_factor > 0:
        all_query_predictions = all_query_predictions[:-(query_pad // args.scale_factor)]  # omit predictions for padded region at end of query
        
    return all_query_predictions

def inference_dcase(model, args, audio_fp, annotations_fp):
    print(f"Inference for {audio_fp}")
    
    fn = os.path.basename(audio_fp)
    
    # loading for speedup
    np_fp = os.path.join(args.experiment_dir, fn[:-4]+".npy")
    if False: #os.path.exists(np_fp):
        audio = load_audio(audio_fp, args.sr)
        annotations = pd.read_csv(annotations_fp)

        support_audio, support_annotations, query_audio, time_shift_sec, min_vox_dur_support, vox_durs_support = process_dcase(audio, annotations, args.sr, args.inference_chunk_size_sec, args.inference_n_chunks_to_keep)
        
        all_query_predictions = np.load(np_fp)
        
    else:        
        model = model.to(device)
        model.eval()
        
        audio = load_audio(audio_fp, args.sr)
        annotations = pd.read_csv(annotations_fp)
        
        if args.inference_hard_negative_sampling:
            # look for high prob negative events in full support audio
            print(f"Applying hard negative sampling")
            support_audio, support_annotations, query_audio, time_shift_sec, min_vox_dur_support, vox_durs_support = process_dcase(audio, annotations, args.sr, args.inference_chunk_size_sec, 1)
            support_audio_full = audio[:int(time_shift_sec*args.sr)]
            support_audio_full_predictions = get_probs(model, support_audio, support_annotations, support_audio_full, args)
            
            # re-generate support audio, incorporating high prob negative events as hard negatives
            print("Re-computing support audio using hard negatives")
            support_audio, support_annotations, query_audio, time_shift_sec, min_vox_dur_support, vox_durs_support = process_dcase(audio, annotations, args.sr, args.inference_chunk_size_sec, args.inference_n_chunks_to_keep, scale_factor=args.scale_factor, support_probs = torch.from_numpy(support_audio_full_predictions))
            all_query_predictions = get_probs(model, support_audio, support_annotations, query_audio, args)
            
        else:        
            support_audio, support_annotations, query_audio, time_shift_sec, min_vox_dur_support, vox_durs_support = process_dcase(audio, annotations, args.sr, args.inference_chunk_size_sec, args.inference_n_chunks_to_keep)
            
            all_query_predictions = get_probs(model, support_audio, support_annotations, query_audio, args)
        
        # save np array of predictions
        fn = os.path.basename(audio_fp)
        np.save(os.path.join(args.experiment_dir, fn[:-4]+".npy"), all_query_predictions)
    d = postprocess(all_query_predictions, audio_fp, args, time_shift_sec, min_vox_dur_support, vox_durs_support)
    
    # Save raven st
    st = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : []}
    for i, row in d.iterrows():
        st["Begin Time (s)"].append(row["Starttime"])
        st["End Time (s)"].append(row["Endtime"])
        st["Annotation"].append("POS")
    st = pd.DataFrame(st)
    st.to_csv(os.path.join(args.experiment_dir, fn[:-4]+'.txt'), index=False, sep='\t')
    
    return d
    