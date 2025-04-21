import pandas as pd
import os
import shutil
import numpy as np

df = pd.read_csv('/home/jupyter/data/fewshot_data/data_large/animalspeak_pseudovox_with_birdnet_with_qf_with_c.csv')
df['pseudovox_audio_fp'] = df['pseudovox_audio_fp'].map(lambda x: x.replace('davidrobinson', 'jupyter/data'))

out_dir = '/home/jupyter/data/fewshot_data/data_large/animalspeak_pseudovox_cluster_review'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

n_to_sample=100

for cc in [f'c_{x}' for x in [8,16,32,64,128]]:
    rng = np.random.default_rng(0)
    
    # Identify clusters with valid sizes
    counts = df[cc].value_counts()
    counts = counts[
        (counts >= 4) & 
        (counts <= 256)
    ]

    allowed_clusters = pd.Series(counts.index)
    to_use = rng.permutation(allowed_clusters)[:n_to_sample]
    
    for c in to_use:
        target_dir = os.path.join(out_dir, cc, str(c))
        os.makedirs(target_dir)
        df_sub = df[df[cc] == c]
        fps = df_sub['pseudovox_audio_fp']
        for fp in list(fps):
            shutil.copy(fp, target_dir)
    