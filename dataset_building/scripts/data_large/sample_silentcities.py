import pandas as pd
from plumbum import local, FG
import os
import numpy as np
import shutil
from tqdm import tqdm
from glob import glob

gcp_filelist = 'silentcities_zipfiles.txt'

temp_dir = '/home/jupyter/silentcities_temp'
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir)

out_dir = '/home/jupyter/silentcities_sampled'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

rng = np.random.default_rng(0)
    
gcp_filelist = pd.read_csv(gcp_filelist, names = ['files'])
gcp_filelist = rng.permutation(sorted(gcp_filelist['files']))

for i, gcp_fp in tqdm(enumerate(gcp_filelist[250:500])):
    local['gsutil']['-m', 'cp', '-r', gcp_fp, temp_dir] & FG
    fn_to_unzip = gcp_fp.split('/')[-1]
    fn_to_unzip = os.path.join(temp_dir, fn_to_unzip)
    local['tar']['-xvf', fn_to_unzip, '-C', temp_dir] & FG
    files = sorted(glob(os.path.join(temp_dir, "*.flac")))
    
    if len(files) > 15:
        files_to_save = rng.choice(files, 15)
    else:
        files_to_save = files
    
    for file_to_save in files_to_save:
        shutil.copy(file_to_save, out_dir)
                        
    shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)