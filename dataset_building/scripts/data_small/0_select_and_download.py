import pandas as pd
import os

all_xeno_canto_files_fp = '/home/jupyter/fewshot/dataset_building/Xeno-canto_filelist.txt'
all_xeno_canto_files = pd.read_csv(all_xeno_canto_files_fp, header=None)

data_small_dir = '/home/jupyter/fewshot/data/data_small'
if not os.path.exists(data_small_dir):
    os.makedirs(data_small_dir)

    
data_small_filelist_fp = '/home/jupyter/fewshot/data/data_small/data_small_filelist.txt'
subselect_xeno_canto_files = all_xeno_canto_files.sample(n=10000, random_state=111)
subselect_xeno_canto_files.to_csv(data_small_filelist_fp, index=False, header=False)

data_small_audio_dir = os.path.join(data_small_dir, "audio")
if not os.path.exists(data_small_audio_dir):
    os.makedirs(data_small_audio_dir)
    
os.system(f'cat {data_small_filelist_fp} | gsutil -m cp -I {data_small_audio_dir}')