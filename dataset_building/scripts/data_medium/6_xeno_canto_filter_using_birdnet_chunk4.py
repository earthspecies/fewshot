from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
from glob import glob
import soundfile as sf
import numpy as np
import os
import shutil
import pandas as pd
from tqdm import tqdm

dataset_name = "xeno_canto"

chunk = 4
chunk_size = 100000

PSEUDOVOX_MANIFEST = f'/home/jupyter/data/fewshot_data/data_medium/{dataset_name}_pseudovox.csv'
TARGET_FP = f'/home/jupyter/data/fewshot_data/data_medium/{dataset_name}_pseudovox_with_birdnet_chunk{chunk}.csv'

df = pd.read_csv(PSEUDOVOX_MANIFEST)

df = df.iloc[chunk*chunk_size:(chunk+1)*chunk_size].reset_index()

# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

probs = []
top_sps = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    
    audio, sr = sf.read(row['pseudovox_audio_fp'])
    
    pad_len = int(.75 * sr)
    audio_padded = np.pad(audio, (pad_len, pad_len))
    
    sf.write(f'temp{chunk}.wav', audio_padded, sr)

    recording = Recording(
        analyzer,
        f'temp{chunk}.wav',
        # lat=35.4244,
        # lon=-120.7463,
        # date=datetime(year=2022, month=5, day=10), # use date or week_48
        min_conf=0,
    )
    recording.analyze()
    
    if len(recording.detections) == 0:
        probs.append(0)
        top_sps.append("")
        
    else:
        top_prob = 0
        top_sp = ""
        for x in recording.detections:
            if x['confidence']>top_prob:
                top_prob = x['confidence']
                top_sp = x['scientific_name']
        probs.append(top_prob)
        top_sps.append(top_sp)
        
    df["birdnet_confidence"] = pd.Series(probs)
    df["birdnet_prediction"] = pd.Series(top_sps)
    
df.to_csv(TARGET_FP, index=False)
    