from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
from glob import glob
import soundfile as sf
import numpy as np
import os
import shutil
import pandas as pd

PSEUDOVOX_MANIFEST = '/home/jupyter/fewshot/data/data_small/pseudovox_manifest_with_clusters.csv'
TARGET_FP = '/home/jupyter/fewshot/data/data_small/pseudovox_manifest_birdnet_filtered.csv'

df = pd.read_csv(PSEUDOVOX_MANIFEST)

# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

probs = []
top_sps = []

for i, row in df.iterrows():
    
    audio, sr = sf.read(row['filepath'])
    
    pad_len = int(.75 * sr)
    audio_padded = np.pad(audio, (pad_len, pad_len))
    
    sf.write('temp.wav', audio_padded, sr)

    recording = Recording(
        analyzer,
        'temp.wav',
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
    