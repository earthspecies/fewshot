'''
This is just to create a list of audio files + durations that can get used as background sounds

This also combines the pseudovox manifest with the clustering info
'''

import pandas as pd
import soundfile as sf
from glob import glob
from tqdm import tqdm
import os

# Background audio

AUDIO_FOLDER_PROCESSED = '/home/jupyter/fewshot/data/data_small/audio_trimmed/'
TARGET_FP = '/home/jupyter/fewshot/data/data_small/background_audio_info.csv'

audio_files = sorted(glob(os.path.join(AUDIO_FOLDER_PROCESSED, '*.wav')))

durs = []

for fp in tqdm(audio_files):
    d = sf.info(fp).duration
    durs.append(d)
    
    
df = pd.DataFrame({'raw_audio_fp' : audio_files, 'duration_sec' : durs})

df.to_csv(TARGET_FP, index=False)

# Merge pseudovox infos

PSEUDOVOX_MANIFEST_FP = '/home/jupyter/fewshot/data/data_small/pseudovox_manifest_edited.csv'
PSEUDOVOX_PREDICTIONS_FP = '/home/jupyter/fewshot/data/data_small/pseudovox_to_clusters.csv'
TARGET_FP = '/home/jupyter/fewshot/data/data_small/pseudovox_manifest_with_clusters.csv'

pseudovox_manifest = pd.read_csv(PSEUDOVOX_MANIFEST_FP)
pseudovox_predictions = pd.read_csv(PSEUDOVOX_PREDICTIONS_FP)

pseudovox_predictions = pseudovox_predictions.drop('filename', axis = 1)

pseudovox_manifest = pseudovox_manifest.merge(pseudovox_predictions, on = 'filepath')

pseudovox_manifest.to_csv(TARGET_FP, index=False)