import pandas as pd
import os

info_fp = '/home/jupyter/data/fewshot_data/data_large/animalspeak_pseudovox_with_birdnet.csv'
qf_fp = '/home/jupyter/data/fewshot_data/data_large/quality_filtered.csv'

info_df = pd.read_csv(info_fp)
qf_df = pd.read_csv(qf_fp)

qf_dict = {}
for i, row in qf_df.iterrows():
    qf_dict[row['filepath']] = row['exclude']
    
def apply_qf(x):
    if x in qf_dict:
        return qf_dict[x]
    else:
        print(f"key {x} not found")
        return True
    
info_df['qf_exclude'] = info_df['pseudovox_audio_fp'].map(apply_qf)

info_df.to_csv(info_fp.replace('.csv', '_with_qf.csv'), index=False)