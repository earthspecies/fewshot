import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import os
from tqdm import tqdm
from einops import rearrange

from fewshot.data.data import get_inference_dataloader, load_audio

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def process_dcase(audio, annotations, args):
    # TODO: UNK is not handled carefully
    # TODO: How to handle extremely long support?
    # print("UNK ASSUMED POS!!")
    
    rng = np.random.default_rng(0)
        
    annotations = annotations.sort_values(by="Starttime").reset_index()
        
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
    # chunk_size_sec = args.support_dur_sec
    chunk_size_samples = int(args.sr * args.inference_chunk_size_sec)
    max_n_chunks_to_keep = args.inference_n_chunks_to_keep
    
    chunks_to_keep = []
    chunks_to_maybe_keep = []
    
    for chunk_start in np.arange(0, support_end_sample, chunk_size_samples):
        chunk_end = min(chunk_start+chunk_size_samples, support_end_sample)
        annot_pos_start_sub = annot_pos[(annot_pos['Starttime'] >= chunk_start/args.sr) & (annot_pos['Starttime'] < chunk_end/args.sr)]
        annot_pos_end_sub = annot_pos[(annot_pos['Endtime'] >= chunk_start/args.sr) & (annot_pos['Endtime'] < chunk_end/args.sr)]
        annot_pos_long_sub = annot_pos[(annot_pos['Starttime'] < chunk_start/args.sr) & (annot_pos['Endtime'] >= chunk_end/args.sr)]
        if len(annot_pos_start_sub) + len(annot_pos_end_sub) + len(annot_pos_long_sub)>0:
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
    
    query_audio = audio[support_end_sample:]
            
    assert len(support_audio) == len(support_annotations)
    
    print(f"Support audio, before subsampling: {support_end_time}. After subsampling: {len(support_audio)/args.sr}")
    
    min_vox_dur_support = (annot_pos_support["Endtime"] - annot_pos_support["Starttime"]).min()
    
    return support_audio, support_annotations, query_audio, support_end_time, min_vox_dur_support

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
    

def postprocess(all_query_predictions, audio_fp, args, time_shift_sec, min_vox_dur_support, threshold=None):
    pred_sr = args.sr // args.scale_factor
    audio_fn = os.path.basename(audio_fp)
    
    if threshold == None:
        all_query_predictions_binary = all_query_predictions >= args.inference_threshold # TODO
    else:
        all_query_predictions_binary = all_query_predictions >= threshold
    
    # fill gaps and omit extremely short predictions
    max_hole_size_sec = np.clip(0.5*min_vox_dur_support, 0.2, 1)
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

def cache_support_encoded(model, support_audio):
    
    # pad and normalize audio
    support_pad_len = (model.audio_chunk_size_samples - support_audio.size(1) % model.audio_chunk_size_samples) % model.audio_chunk_size_samples
    if support_pad_len>0:
        support_audio = F.pad(support_audio, (0,support_pad_len))
        
    normalization_factor = torch.std(support_audio, dim=1, keepdim=True)
    normalization_factor = torch.maximum(normalization_factor, torch.full_like(normalization_factor, 1e-6))
    support_audio = (support_audio - torch.mean(support_audio, dim=1,keepdim=True)) / normalization_factor
    
    support_len_samples = support_audio.size(1)
    
    support_audio_subs_encoded = []
    start_samples = []
    
    for start_sample in range(0, support_len_samples, model.audio_chunk_size_samples):
        support_audio_sub = support_audio[:, start_sample:start_sample+model.audio_chunk_size_samples]
        support_audio_sub_encoded = model.audio_encoder(support_audio_sub)
        support_audio_subs_encoded.append(support_audio_sub_encoded)
        start_samples.append(start_sample)
        
    return support_audio_subs_encoded, start_samples


def forward_cached(model, support_audio, support_audio_encoded, start_samples, support_labels, query_audio, query_labels=None, temperature=1):
    """
    Input
        support_audio (Tensor): (batch, time) (at audio_sr)
        support_labels (Tensor): (batch, time) (at audio_sr)
        query_audio (Tensor): (batch, time) (at audio_sr)
        query_labels (Tensor): (batch, time) (at audio_sr)
    Output
        logits (Tensor): (batch, query_time/scale_factor) (at audio_sr / scale factor)
    """

    # pad and normalize audio
    support_pad_len = (model.audio_chunk_size_samples - support_audio.size(1) % model.audio_chunk_size_samples) % model.audio_chunk_size_samples
    if support_pad_len>0:
        support_labels = F.pad(support_labels, (0,support_pad_len))

    normalization_factor = torch.std(support_audio, dim=1, keepdim=True)
    normalization_factor = torch.maximum(normalization_factor, torch.full_like(normalization_factor, 1e-6))
    query_audio = (query_audio - torch.mean(query_audio, dim=1,keepdim=True)) / normalization_factor

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
    
    query_outputs = model.confidence_transformer(query_confidences)[:,0,:] # bt c
    query_outputs = torch.reshape(query_outputs, (c_shape[0], c_shape[1], -1)) # b t c

    query_logits = model.detection_head(query_outputs).squeeze(2)
    # query_denoised_spec_prediction = self.denoising_head(query_outputs)
    # query_denoised_spec_prediction = rearrange(query_denoised_spec_prediction, 'b c t -> b t c')

    if query_labels is not None:
        query_labels = torch.unsqueeze(query_labels, 1) # (batch, 1, time)
        query_labels = F.max_pool1d(query_labels, model.args.scale_factor, padding=0) # (batch, 1 , time/scale_factor). 0=NEG 1=UNK 2=POS
        query_labels = torch.squeeze(query_labels, 1) # (batch, time/scale_factor)

    return query_logits, query_labels

def inference_dcase(model, args, audio_fp, annotations_fp):
    print(f"Inference for {audio_fp}")
    
    fn = os.path.basename(audio_fp)
    
    # loading for speedup
    np_fp = os.path.join(args.experiment_dir, fn[:-4]+".npy")
    if os.path.exists(np_fp):
        audio = load_audio(audio_fp, args.sr)
        annotations = pd.read_csv(annotations_fp)

        support_audio, support_annotations, query_audio, time_shift_sec, min_vox_dur_support = process_dcase(audio, annotations, args)
        all_query_predictions = np.load(np_fp)
    #
        
    else:
        model = model.to(device)
        model.eval()

        audio = load_audio(audio_fp, args.sr)
        annotations = pd.read_csv(annotations_fp)

        support_audio, support_annotations, query_audio, time_shift_sec, min_vox_dur_support = process_dcase(audio, annotations, args)
        
        # pad etc
        
        ## loop audio at offset to provide a windowing effect
        support_training_dur_samples = int(args.sr*args.support_dur_sec)
        support_dur_samples = support_audio.size(0)
        
        assert model.audio_chunk_size_samples % 2 == 0
        assert support_dur_samples >= model.audio_chunk_size_samples//2
        
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
        
        ##
        support_dataloader = get_inference_dataloader(support_audio, support_annotations, support_audio, args)
        all_support_predictions = []
        ##
        
        if args.inference_temperature <0:
            temperature= (args.support_dur_sec//args.audio_chunk_size_sec)/len(inference_dataloader) # adjust from training to inference by decreasing temp for long clips
        else:
            temperature=args.inference_temperature
        with torch.no_grad():
            cached_support_encoded = None
            for i, data_item in tqdm(enumerate(inference_dataloader)):
                sa, sl, qa = data_item
                sa = sa.to(device)
                
                if cached_support_encoded is None:
                    cached_support_encoded, start_samples = cache_support_encoded(model, sa.to(device))
                
                query_predictions, _ = forward_cached(model, sa.to(device), cached_support_encoded, start_samples, sl.to(device), qa.to(device), temperature=temperature) # lower temperature gives more weight to high confidence votes; use as hyperparam for long support sets?
                all_query_predictions.append(query_predictions)
        
        
        all_query_predictions = torch.cat(all_query_predictions, dim=0)
        assert all_query_predictions.size(1) % 4 == 0
        quarter_window = all_query_predictions.size(1)//4
        
        all_query_predictions_windowed = torch.reshape(all_query_predictions[:, quarter_window:-quarter_window], (-1,))
        all_query_predictions = torch.cat((all_query_predictions[0,:quarter_window], all_query_predictions_windowed, all_query_predictions[-1,-quarter_window:])).cpu().numpy()
        
        #remove query padding        
        if query_pad // args.scale_factor > 0:
            all_query_predictions = all_query_predictions[:-(query_pad // args.scale_factor)] # omit predictions for padded region at end of query
        
#         ##
#         with torch.no_grad():
#             for i, data_item in tqdm(enumerate(support_dataloader)):
#                 sa, sl, sqa = data_item
#                 sa = sa.to(device)
                
#                 support_predictions, support_labels_downsampled = forward_cached(model, sa.to(device), cached_support_encoded, start_samples, sl.to(device), sqa.to(device), temperature=temperature, query_labels=sl.to(device)) # lower temperature gives more weight to high confidence votes; use as hyperparam for long support sets?
#                 all_support_predictions.append(support_predictions)
        
#         all_support_predictions = torch.cat(all_support_predictions, dim=0)
#         assert all_support_predictions.size(1) % 4 == 0
#         quarter_window = all_support_predictions.size(1)//4
        
#         all_support_predictions_windowed = torch.reshape(all_support_predictions[:, quarter_window:-quarter_window], (-1,))
#         all_support_predictions = torch.cat((all_support_predictions[0,:quarter_window], all_support_predictions_windowed, all_support_predictions[-1,-quarter_window:])).cpu().numpy()   
        
#         support_labels_downsampled = support_labels_downsampled[0,:].cpu().numpy()
            
#         fn = os.path.basename(audio_fp)
#         np.save(os.path.join(args.experiment_dir, fn[:-4]+"_support_predictions.npy"), all_support_predictions)
                
#         support_pos = all_support_predictions[support_labels_downsampled ==2]
#         support_neg = all_support_predictions[support_labels_downsampled ==0]
        
#         avg_pos = np.median(support_pos)
#         avg_neg = np.median(support_neg)
#         print(avg_pos)
        
# #         threshold_learned = (avg_pos+avg_neg)/2
        
#         threshold_learned = avg_pos - 0.1*(avg_pos-avg_neg)
#         # threshold_learned=0
#         print(threshold_learned)
        
#         ##
        
        # save np array of predictions
        fn = os.path.basename(audio_fp)
        np.save(os.path.join(args.experiment_dir, fn[:-4]+".npy"), all_query_predictions)
    
    d = postprocess(all_query_predictions, audio_fp, args, time_shift_sec, min_vox_dur_support)
    
    # Save raven st
    st = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : []}
    for i, row in d.iterrows():
        st["Begin Time (s)"].append(row["Starttime"])
        st["End Time (s)"].append(row["Endtime"])
        st["Annotation"].append("POS")
    st = pd.DataFrame(st)
    st.to_csv(os.path.join(args.experiment_dir, fn[:-4]+'.txt'), index=False, sep='\t')
    
    return d
    