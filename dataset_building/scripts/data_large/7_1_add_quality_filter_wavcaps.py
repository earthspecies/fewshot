import pandas as pd
import os
from tqdm import tqdm

info_fp = '/home/jupyter/data/fewshot_data/data_large/wavcaps_pseudovox_with_birdnet_wembeddings.csv'
qf_fp = '/home/jupyter/data/fewshot_data/data_large/wavcaps_quality_filtered.csv'

info_df = pd.read_csv(info_fp)
qf_df = pd.read_csv(qf_fp)

qf_dict = {}
for i, row in tqdm(qf_df.iterrows(), total=len(qf_df)):
    qf_dict[row['filepath']] = row['exclude']
    
def apply_qf(x):
    if x in qf_dict:
        return qf_dict[x]
    else:
        print(f"key {x} not found")
        return True
    
tqdm.pandas()
    
info_df['qf_exclude'] = info_df['pseudovox_audio_fp'].progress_map(apply_qf)

info_df.to_csv(info_fp.replace('_wembeddings.csv', '_with_qf.csv'), index=False)