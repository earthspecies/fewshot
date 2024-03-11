'''
Create csv required by clustering module (currently belsuono)

After running belsuono, outputs should be copied to data directory
'''

import pandas as pd
import os

pseudovox_df = pd.read_csv('/home/jupyter/fewshot/data/data_small/pseudovox.csv')

pseudovox_df['filename'] = pseudovox_df['pseudovox_audio_fp'].map(lambda x : os.path.basename(x).replace('.wav', ''))
pseudovox_df['filepath'] = pseudovox_df['pseudovox_audio_fp']

# some pseudovox are too long

longest_allowed_pseudovox_sec = 6

pseudovox_df = pseudovox_df[pseudovox_df['End Time (s)'] - pseudovox_df['Begin Time (s)'] < longest_allowed_pseudovox_sec]

pseudovox_df.to_csv('/home/jupyter/fewshot/data/data_small/pseudovox_manifest_edited.csv', index=False)

'''
After this, from belsuono I ran:

python run.py --expt-name=fewshot_data_small_avg50_v2 --dataset-info-fp=/home/jupyter/fewshot/data/data_small/pseudovox_manifest_edited.csv --features=avesonly --clustering=minibatch_kmeans --n-clusters=3620 --num-workers=0

'''