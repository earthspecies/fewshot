import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import json
import yaml

import torch
import torchaudio
from einops import rearrange


spectrogram = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=320)

def get_snr(audio_fp, anno_fp, highfreq, lowfreq, sr=16000):
    freq_bins = torch.fft.rfftfreq(2048, d=1/sr) # torch.arange((n + 1) // 2) / (d * n)
    st = pd.read_csv(anno_fp)
    audiofilename = list(st["Audiofilename"])[0]

    st_sub = st[st["Q"] != "UNK"]
    support_endtime = sorted(st_sub["Endtime"])[4]
    
    support_st = st_sub[st_sub["Endtime"] <= support_endtime]
    
    support_endsample = int(support_endtime * sr)+1
    
    audio, _ = librosa.load(audio_fp, duration=support_endtime, mono=True, sr=sr)
    audio = torch.tensor(audio)
    
    labels = torch.zeros_like(audio)

    ambient_rms = max(1e-10, float(torch.std(audio)))

    support_st = support_st.sort_values('Endtime')

    for i, row in support_st.iterrows():
        begin_sample = int(row['Starttime'] * sr)
        end_sample = min(audio.size(0),max(begin_sample+1, int(row['Endtime'] * sr)))
        l = {"POS" : 2, "UNK" : 1, "NEG" : 0}[row["Q"]]
        labels[begin_sample:end_sample] = torch.maximum(labels[begin_sample:end_sample], torch.full_like(labels[begin_sample:end_sample], l))


#     st_sub_support = st_sub[st_sub["Endtime"] <= support_endtime]
#     assert len(st_sub_support) == self.n_shots
#     self.min_support_vox_dur = (st_sub_support["Endtime"] - st_sub_support["Starttime"]).min()

#     self.support_endtime = support_endtime
#     self.query_starttime = query_starttime

#     self.support_audio = audio[:support_endsample]
#     self.support_labels = labels[:support_endsample]

#     self.query_audio = audio[query_startsample:]
#     self.query_labels = labels[query_startsample:]

    # Get band-limited snr
    event_spec = spectrogram(audio[labels>0])
    event_spec = rearrange(event_spec, 'c t -> t c').unsqueeze(0)
    background_audio = audio[labels==0]
    background_audio_sub = background_audio[:sr*10]
    background_spec = spectrogram(background_audio_sub)
    background_spec = rearrange(background_spec, 'c t -> t c').unsqueeze(0)

    # For frame_number, outputs 1 (detection) or 0 (no detection) based on band-limited energy, as compared to noise floor
    freq_bins = freq_bins.unsqueeze(0).unsqueeze(0)
    freq_bins_mask = (freq_bins>=lowfreq) * (freq_bins<=highfreq)

    event_freq_bins_mask = torch.tile(freq_bins_mask, (1,event_spec.size(1),1))
    event_spec_bandlim = event_spec * event_freq_bins_mask
    event_energy = event_spec_bandlim.sum(dim=-1)
    event_energy_mean = event_energy.mean(dim=-1)

    background_freq_bins_mask = torch.tile(freq_bins_mask, (1,background_spec.size(1),1))
    background_spec_bandlim = background_spec * background_freq_bins_mask
    background_energy = background_spec_bandlim.sum(dim=-1)# [batch,time]

    twenty_percent_noise_estimate = torch.quantile(background_energy, q=0.2, dim = -1)

    snr_db = 10.*(torch.log10(event_energy_mean) - torch.log10(twenty_percent_noise_estimate)).squeeze()
    snr_db = torch.nan_to_num(snr_db, posinf=50, neginf=-50)
    
    return float(snr_db)

results_fp = '/home/jupyter/drasdic/results/drasdic/Evaluation_report_eval_random_9_80000_model_X_20012025_06_08_36.json'
with open(results_fp, 'r') as f:
    results = json.load(f)
    
precomputed_features_fp = '/home/jupyter/data/fewshot_data/evaluation/bboxes/precomputed.yaml'
with open(precomputed_features_fp, 'r') as f:
    precomputed_features = yaml.safe_load(f)

datasets = ["marmoset_halftime", "Anuraset", "carrion_crow", "katydid_sixthtime", "Spiders_halftime", "rana_sierrae", "Powdermill",  "Hawaii", "right_whale", "gibbons", "gunshots", "humpback", "ruffed_grouse"]

dataset_to_dataset_name = {"marmoset_halftime" : "MS",
                           "Anuraset" : "AS",
                           "carrion_crow" : "CC",
                           "katydid_sixthtime" : "KD",
                           "Spiders_halftime" : "JS", 
                           "rana_sierrae" : "RS", 
                           "Powdermill" : "PM", 
                           "Hawaii" : "HA",
                           "right_whale" : "RW", 
                           "gibbons" : "HG",
                           "gunshots" : "GS", 
                           "humpback" : "HW",
                           "ruffed_grouse" : "RG"
                          }

formatted_dataset_parent_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted_for_dcase"

features = ["Duration (sec)", "Rate (events/sec)", "High Freq (Hz)", "Low Freq (Hz)", "SNR (dB)"]

allresults = {"audiofile" : [], "F1" : [], "dataset" : []}
for feature in features:
    allresults[feature] = []

for audiofile in tqdm(sorted(results['scores_per_audiofile'].keys())):
    keep = False
    for dataset in datasets:
        if (dataset == audiofile[:len(dataset)]) and ("crossfile" not in audiofile):
            keep = True
            current_dataset = dataset
            break
    if not keep:
        continue
        
    score = results['scores_per_audiofile'][audiofile]["f-measure"]
    
    allresults["audiofile"].append(audiofile)
    allresults["F1"].append(score)
    allresults["dataset"].append(dataset_to_dataset_name[current_dataset])
    
    audio_fp = os.path.join(formatted_dataset_parent_dir, current_dataset, audiofile)
    st_fp = os.path.join(formatted_dataset_parent_dir, current_dataset, audiofile.replace('.wav', '.csv'))
    st_fn = os.path.basename(st_fp)
    st = pd.read_csv(st_fp)
    st = st[st["Q"] == "POS"].sort_values("Endtime").reset_index(drop=True).copy()
    
    for feature in features:
        ## Duration
        if feature == "Duration (sec)":
            st[feature] = st['Endtime'] - st['Starttime']
            durmedian = st[feature].median()
            allresults[feature].append(durmedian)
            
        if feature == "Rate (events/sec)":
            n_events = len(st)
            n_seconds = librosa.get_duration(path=audio_fp)
            allresults[feature].append(n_events/n_seconds)
            
        if feature in ["High Freq (Hz)", "Low Freq (Hz)"]:
            x = precomputed_features[st_fn][feature]
            x = min(8000,x)
            x = max(10, x)
            allresults[feature].append(x)
            
        if feature == "SNR (dB)":
            highfreq = precomputed_features[st_fn]["High Freq (Hz)"]
            lowfreq = precomputed_features[st_fn]["Low Freq (Hz)"]
            snr = get_snr(audio_fp, st_fp, highfreq, lowfreq, sr=16000)
            allresults[feature].append(snr)
            
#         if feature == "Peak Freq (Hz)":
#             st_sub = st.iloc[:5]
#             assert len(st_sub) == 5
#             st
        
allresults = pd.DataFrame(allresults).sort_values("dataset")
for feature in features:
    abbrev = feature.split(' ')[0]
    outfp = f'/home/jupyter/drasdic/results/{abbrev}_score_summary.png'
    sns.scatterplot(data=allresults, x=feature, y="F1", hue="dataset")
    
    if feature not in ["SNR (dB)"]:
        plt.xscale("log")
    
    plt.savefig(outfp)
    plt.close()

