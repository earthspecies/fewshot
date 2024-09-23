import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import patches
import seaborn as sns

datasets = ["marmoset", "Anuraset", "carrion_crow", "katydid", "Spiders", "rana_sierrae", "Powdermill",  "Hawaii", "right_whale", "gibbons", "gunshots", "humpback", "ruffed_grouse", "audioset_strong", "DESED"]
dataset_names = [x.replace('_', ' ').title() for x in datasets]
formatted_dataset_parent_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted"

fig, ax = plt.subplots(len(datasets),3, figsize=(20, 5*len(datasets)))

for i, (dataset_name, dataset) in enumerate(sorted(zip(dataset_names, datasets))):
    print(f"Processing {dataset}")
    
    data_dir = os.path.join(formatted_dataset_parent_dir, dataset)
    manifest = pd.read_csv(os.path.join(data_dir, "manifest.csv"))
    
    example_files = manifest.sample(3, random_state = 2, replace=True)
    for filen in range(3):
        audio_fp = os.path.join(formatted_dataset_parent_dir, list(example_files["audio_fp"])[filen])

        st_fp = os.path.join(formatted_dataset_parent_dir, list(example_files["selection_table_fp"])[filen])

        st = pd.read_csv(st_fp, sep='\t')
        st = st[st["Annotation"]!="Unknown"]
        st_sub = st[st["Begin Time (s)"] > 2]
        example_start = list(st_sub["Begin Time (s)"].sample(1, random_state=filen))[0]-1

        y, sr = librosa.load(audio_fp, offset=example_start, duration = 10, sr = None)
        
#         print(audio_fp)
#         print(sr)

        fmax = min(10000, sr//2)
        hop_length = 1024
        D = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=fmax, fmin=50), ref=np.max)
        # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='mel', x_axis='time', sr=sr, ax=ax[i, filen], fmax=fmax, fmin=100)
        
        st_sub = st[((st["Begin Time (s)"] >= example_start) & (st["Begin Time (s)"] <= example_start+10)) | ((st["End Time (s)"] >= example_start) & (st["End Time (s)"] <= example_start+10))]
        for _, row in st_sub.iterrows():
            boxstart = max(0, row["Begin Time (s)"] - example_start)
            boxend = min(row["End Time (s)"] - example_start, 10)
            
            ax[i,filen].axvspan(boxstart, boxend, color='white', alpha=0.2, label='Event of Interest', linewidth=2)
        
        if filen == 1:
            ax[i, filen].set(title=dataset_name)

fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.savefig(os.path.join(formatted_dataset_parent_dir, "example_audio.png"))
    