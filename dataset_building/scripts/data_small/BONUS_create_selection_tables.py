import pandas as pd
import numpy as np
import os
from tqdm import tqdm

'''
script to create selection tables for the original audio files, for all the sounds that ended up being pseudovox.

potentially to use for training voxaboxen as a general purpose region proposal network
'''

AUDIO_MANIFEST_FP = '/home/jupyter/fewshot/data/data_small/pseudovox_manifest_birdnet_filtered.csv'
OUTPUT_DIR = '/home/jupyter/fewshot/data/data_small/pseudovox_selection_tables'
BIRDNET_CONFIDENCE_STRICT_LOWER_BOUND = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(AUDIO_MANIFEST_FP)

original_fps = df['raw_audio_fp'].unique()

for original_fp in tqdm(original_fps):
    df_sub = df[df['raw_audio_fp'] == original_fp]
    df_sub = df_sub[df_sub['birdnet_confidence'] > BIRDNET_CONFIDENCE_STRICT_LOWER_BOUND]
    
    st = pd.DataFrame({'Begin Time (s)' : df_sub['Begin Time (s)'], 'End Time (s)' : df_sub['End Time (s)'], 'Annotation' : df_sub['birdnet_prediction']})
    
    original_fn = ".".join(os.path.basename(original_fp).split('.')[:-1])
    target_fn = f"pseudovox_selections_{original_fn}.txt"
    
    target_fp = os.path.join(OUTPUT_DIR, target_fn)
    st.to_csv(target_fp, index=False, sep='\t')
