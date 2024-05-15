import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import os
from tqdm import tqdm
from einops import rearrange

from fewshot.data.data import get_inference_dataloader, load_audio
from fewshot.models.aves_plus_labelencoder.params import load_params
from fewshot.models.aves_plus_labelencoder.model import FewShotModel
from fewshot.models.aves_plus_labelencoder.inference import postprocess, cache_support_encoded, forward_cached

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def infer(model_checkpoint_fp, model_args_fp, support_audio_fp, support_selection_table_fp, query_audio_fp, out_fp=None):
    args = load_params(model_args_fp)
    
    print("Init model")
    model=FewShotModel(args)
    
    print(f"loading model weights from {model_checkpoint_fp}")
    cp = torch.load(model_checkpoint_fp)
    model.load_state_dict(cp["model_state_dict"])
    
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    support_audio = load_audio(support_audio_fp, args.sr)
    query_audio = load_audio(query_audio_fp, args.sr)
    support_st = pd.read_csv(support_selection_table_fp, sep='\t')
    
    support_annotations = torch.zeros_like(support_audio)
    for i, row in support_st.iterrows():
        begin_sample = int(row['Begin Time (s)'] * args.sr)
        end_sample = int(row["End Time (s)"] * args.sr)
        support_annotations[begin_sample:end_sample] = 2
    
    #
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

    with torch.no_grad():
        cached_support_encoded = None
        for i, data_item in tqdm(enumerate(inference_dataloader)):
            sa, sl, qa = data_item
            sa = sa.to(device)

            if cached_support_encoded is None:
                cached_support_encoded, start_samples = cache_support_encoded(model, sa.to(device))

            query_predictions, _ = forward_cached(model, sa.to(device), cached_support_encoded, start_samples, sl.to(device), qa.to(device), temperature=0) # lower temperature gives more weight to high confidence votes; use as hyperparam for long support sets?
            all_query_predictions.append(query_predictions)

    all_query_predictions = torch.cat(all_query_predictions, dim=0)
    assert all_query_predictions.size(1) % 4 == 0
    quarter_window = all_query_predictions.size(1)//4

    all_query_predictions_windowed = torch.reshape(all_query_predictions[:, quarter_window:-quarter_window], (-1,))
    all_query_predictions = torch.cat((all_query_predictions[0,:quarter_window], all_query_predictions_windowed, all_query_predictions[-1,-quarter_window:])).cpu().numpy()

    #remove query padding        
    if query_pad // args.scale_factor > 0:
        all_query_predictions = all_query_predictions[:-(query_pad // args.scale_factor)] # omit predictions for padded region at end of query
    
    min_vox_dur_support = (support_st['End Time (s)'] - support_st["Begin Time (s)"]).min()
    d = postprocess(all_query_predictions, query_audio_fp, args, 0, min_vox_dur_support, threshold=-0.5)
    
    # Save raven st
    st = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : []}
    for i, row in d.iterrows():
        st["Begin Time (s)"].append(row["Starttime"])
        st["End Time (s)"].append(row["Endtime"])
        st["Annotation"].append("POS")
    st = pd.DataFrame(st)
    
    if out_fp is not None:
        st.to_csv(out_fp, sep='\t', index=False)
        
    return st
    
    
    
