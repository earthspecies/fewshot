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

data_dir = '/home/jupyter/drasdic_demo_clips_nodomainshift'

features = ["Duration (s)", "SNR (dB)", "Peak Frequency (Hz)", "Spectral Flatness (dB)"]

summary = {"Filename" : [], 
           "First Five" : [],
          }

for featname in features:
    summary[featname] = []

support_dur_sec = 60

audio_fps = sorted(glob(os.path.join(data_dir, "*.wav")))

for audio_fp in tqdm(audio_fps):
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

    
    st_fp = audio_fp.replace(".wav", ".txt").replace("/audio_", "/selection_table_")
    st = pd.read_csv(st_fp, sep='\t')

    st = st[st["Annotation"] != "Unknown"]
    st = st.sort_values("Begin Time (s)")

    st["SNR (dB)"] = st.apply(get_snr, axis=1)
    st["Peak Frequency (Hz)"] = st.apply(get_peak_frequency, axis = 1)
    st["Spectral Flatness (dB)"] = st.apply(get_flatness, axis = 1)

    st["Duration (s)"] = st["End Time (s)"] - st["Begin Time (s)"]

#     def get_n_calls_within_60(begin_time):
#         st_sub = st[(st["Begin Time (s)"] - begin_time < 60) & (st["Begin Time (s)"] - begin_time > 0)]
#         return len(st_sub)

#     st["Number events within next 60s"] = st["Begin Time (s)"].map(get_n_calls_within_60)

    begin_time_5 = support_dur_sec #sorted(st["Begin Time (s)"])[4]
    st_5 = st[st["Begin Time (s)"] <= begin_time_5]
    st_after5 = st[st["Begin Time (s)"] > begin_time_5]
    
    if len(st_after5) == 0:
        continue
    # assert len(st_5) == 5

    for featname in features:
        summary[featname].extend(list(st_5[featname]))
    summary["Filename"].extend([audio_fp for _ in range(len(st_5))])
    summary["First Five"].extend([True for _ in range(len(st_5))])

    for featname in features:
        summary[featname].extend(list(st_after5[featname]))
    summary["Filename"].extend([audio_fp for _ in range(len(st_after5))])
    summary["First Five"].extend([False for _ in range(len(st_after5))])
        
summary = pd.DataFrame(summary)

fig, ax = plt.subplots(len(features),1, figsize=(12,12*len(features)))

for i, featname in enumerate(features):
    df_sub = summary
    toplotx = []
    toploty = []
    for fn in sorted(df_sub['Filename'].unique()):
        for firstfive in [True, False]:
            df_sub_sub = df_sub[df_sub['Filename'] == fn]
            df_sub_sub = df_sub_sub[df_sub_sub['First Five'] == firstfive]
            if firstfive:
                toplotx.append(df_sub_sub[featname].mean())
            else:
                toploty.append(df_sub_sub[featname].mean())
    ax[i].scatter(x=toplotx, y=toploty)
    atp = toplotx + toploty
    ax[i].set_ylim(bottom = np.amin(atp, where=np.isfinite(atp), initial=0), top = np.amax(atp, where=np.isfinite(atp), initial=0))
    ax[i].set_xlim(left = np.amin(atp, where=np.isfinite(atp), initial=0), right = np.amax(atp, where=np.isfinite(atp), initial=0))
    
    ax[i].set_title(featname)
    ax[i].set_xlabel("Support (Mean)")
    ax[i].set_ylabel("Remaining (Mean)")
    ax[i].legend()
    
output_fp = os.path.join("/home/jupyter/data/fewshot_data/data_large", "summary2_nodomainshift.png")
plt.savefig(output_fp)
