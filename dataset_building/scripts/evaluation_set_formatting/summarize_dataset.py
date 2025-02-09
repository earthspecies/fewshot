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

datasets = ["marmoset", "Anuraset", "carrion_crow", "katydid", "Spiders", "rana_sierrae", "Powdermill",  "Hawaii", "right_whale", "gibbons", "gunshots", "humpback", "ruffed_grouse"]

dataset_to_dataset_name = {"marmoset" : "MS",
                           "Anuraset" : "AS",
                           "carrion_crow" : "CC",
                           "katydid" : "KD",
                           "Spiders" : "JS", 
                           "rana_sierrae" : "RS", 
                           "Powdermill" : "PM", 
                           "Hawaii" : "HA",
                           "right_whale" : "RW", 
                           "gibbons" : "HG",
                           "gunshots" : "GS", 
                           "humpback" : "HW",
                           "ruffed_grouse" : "RG"
                          }

dataset_names = [dataset_to_dataset_name[x] for x in datasets]
datasets = [x for _,x in sorted(zip(dataset_names, datasets))]

formatted_dataset_parent_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted"

features = ["Duration (s)", "Number events within next 60s", "SNR (dB)", "Peak Frequency (Hz)", "Spectral Flatness (dB)"]

summary = {"Filename" : [], 
           "Dataset" : [],
           "First Five" : [],
          }

for featname in features:
    summary[featname] = []

for dataset in datasets:
    print(f"Processing {dataset}")
    dataset_name = dataset_to_dataset_name[dataset]
    
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

for i, featname in enumerate(features):
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    sns.boxplot(data=summary[~summary["First Five"]], y="Dataset", x=featname, ax = ax, whis=(0, 100), color="white")
    sns.stripplot(data=summary[summary["First Five"]], y="Dataset", x=featname, ax = ax, color="blue", alpha=0.5)
    if featname in ["Peak Frequency (Hz)"]:
        ax.set_xscale('log')
    ax.set_title(f"Per-Dataset Event {featname}")
    
    output_fp = os.path.join("/home/jupyter/data/fewshot_data/evaluation/formatted", f"summary_{featname}.png")
    plt.savefig(output_fp)
    plt.close()



# features2 = ["Duration (s)", "SNR (dB)", "Peak Frequency (Hz)", "Spectral Flatness (dB)"]
# fig, ax = plt.subplots(len(features2),1, figsize=(12,12*len(features2)))

# for i, featname in enumerate(features2):
#     atp = []
#     for dataset in sorted(summary['Dataset'].unique()):
#         df_sub = summary[summary["Dataset"] == dataset]
#         toplotx = []
#         toploty = []
#         for fn in sorted(df_sub['Filename'].unique()):
#             for firstfive in [True, False]:
#                 df_sub_sub = df_sub[df_sub['Filename'] == fn]
#                 df_sub_sub = df_sub_sub[df_sub_sub['First Five'] == firstfive]
#                 if firstfive:
#                     toplotx.append(df_sub_sub[featname].mean())
#                 else:
#                     toploty.append(df_sub_sub[featname].mean())
#         ax[i].scatter(x=toplotx, y=toploty, label=dataset)
#         atp.extend(toplotx)
#         atp.extend(toploty)
#     ax[i].set_ylim(bottom = np.amin(atp, where=np.isfinite(atp), initial=0), top = np.amax(atp, where=np.isfinite(atp), initial=0))
#     ax[i].set_xlim(left = np.amin(atp, where=np.isfinite(atp), initial=0), right = np.amax(atp, where=np.isfinite(atp), initial=0))
    
#     ax[i].set_title(featname)
#     ax[i].set_xlabel("First Five (Mean)")
#     ax[i].set_ylabel("Remaining (Mean)")
#     ax[i].legend()
    
# output_fp = os.path.join("/home/jupyter/data/fewshot_data/evaluation/formatted", "summary2.png")
# plt.savefig(output_fp)
