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

datasets = ["marmoset", "Anuraset", "carrion_crow", "katydid", "Spiders", "rana_sierrae", "Powdermill",  "Hawaii", "right_whale", "gibbons", "gunshots", "humpback", "ruffed_grouse", "audioset_strong", "DESED"]
dataset_names = [x.replace('_', ' ').title() for x in datasets]
formatted_dataset_parent_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted"

features = ["Duration (s)", "Number events within next 60s", "SNR (dB)", "Peak Frequency (Hz)", "Spectral Flatness (dB)"]

summary = {"Filename" : [], 
           "Dataset" : [],
           "First Five" : [],
          }

for featname in features:
    summary[featname] = []

for dataset_name, dataset in sorted(zip(dataset_names, datasets)):
    print(f"Processing {dataset}")
    
    data_dir = os.path.join(formatted_dataset_parent_dir, dataset)
    manifest = pd.read_csv(os.path.join(data_dir, "manifest.csv"))
    
    for i, row in tqdm(manifest.iterrows(), total=len(manifest)):
        audio_fp = os.path.join(formatted_dataset_parent_dir, row["audio_fp"])
        audio, sr = librosa.load(audio_fp, sr=None)
        background_power = np.var(audio)
        
        def get_snr(row):
            start = row["Begin Time (s)"]
            start = int(start*sr)
            end = row["End Time (s)"]
            end = int(end*sr)
            event = audio[start:end]
            power = np.var(event)
            snr = 10*np.log10(power/background_power)
            return snr
        
        def get_peak_frequency(row):
            start = row["Begin Time (s)"]
            start = int(start*sr)
            end = row["End Time (s)"]
            end = int(end*sr)
            event = audio[start:end]
            
            n_fft = 2048
            
            spec = np.abs(librosa.stft(event, n_fft=n_fft))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # remove low frequency bands b/c of artifacts
            freqs = freqs[3:]
            spec = spec[3:, :]
            
            peak_idx = np.argmax(spec, axis = 0)
            
            peak_freqs = [freqs[i] for i in peak_idx]
            
            return np.mean(peak_freqs)
        
        def get_flatness(row):
            start = row["Begin Time (s)"]
            start = int(start*sr)
            end = row["End Time (s)"]
            end = int(end*sr)
            event = audio[start:end]
            
            flatness = librosa.feature.spectral_flatness(y=event, n_fft=2048, hop_length=512)
            flatness = 10*np.log10(flatness)
            
            return np.mean(flatness)
            
            
        st_fp = os.path.join(formatted_dataset_parent_dir, row["selection_table_fp"])
        st = pd.read_csv(st_fp, sep='\t')
        
        st = st[st["Annotation"] != "Unknown"]
        st = st.sort_values("Begin Time (s)")
        
        st["SNR (dB)"] = st.apply(get_snr, axis=1)
        st["Peak Frequency (Hz)"] = st.apply(get_peak_frequency, axis = 1)
        st["Spectral Flatness (dB)"] = st.apply(get_flatness, axis = 1)
        
        st["Duration (s)"] = st["End Time (s)"] - st["Begin Time (s)"]
        
        def get_n_calls_within_60(begin_time):
            st_sub = st[(st["Begin Time (s)"] - begin_time < 60) & (st["Begin Time (s)"] - begin_time > 0)]
            return len(st_sub)
        
        st["Number events within next 60s"] = st["Begin Time (s)"].map(get_n_calls_within_60)
        
        begin_time_5 = sorted(st["Begin Time (s)"])[4]
        st_5 = st[st["Begin Time (s)"] <= begin_time_5]
        st_after5 = st[st["Begin Time (s)"] > begin_time_5]
        assert len(st_5) == 5
        
        for featname in features:
            summary[featname].extend(list(st_5[featname]))
        summary["Filename"].extend([audio_fp for _ in range(len(st_5))])
        summary["Dataset"].extend([dataset_name for _ in range(len(st_5))])
        summary["First Five"].extend([True for _ in range(len(st_5))])
        
        for featname in features:
            summary[featname].extend(list(st_after5[featname]))
        summary["Filename"].extend([audio_fp for _ in range(len(st_after5))])
        summary["Dataset"].extend([dataset_name for _ in range(len(st_after5))])
        summary["First Five"].extend([False for _ in range(len(st_after5))])
        
summary = pd.DataFrame(summary)

fig, ax = plt.subplots(len(features),1, figsize=(16,7*len(features)))

for i, featname in enumerate(features):
    sns.boxplot(data=summary, y="Dataset", x=featname, ax = ax[i], whis=(0, 100), color="white")
    sns.stripplot(data=summary[summary["First Five"]], y="Dataset", x=featname, ax = ax[i], color="blue", alpha=0.5)
    if featname in ["Peak Frequency (Hz)"]:
        ax[i].set_xscale('log')
    
output_fp = os.path.join("/home/jupyter/data/fewshot_data/evaluation/formatted", "summary.png")
plt.savefig(output_fp)
