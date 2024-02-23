import os
from glob import glob
import pandas as pd

data_dir = '/home/jupyter/fewshot/data/data_small'
unseparated_audio_dir = os.path.join(data_dir, "audio")
separated_audio_dir = os.path.join(data_dir, "audio_mixit")

if not os.path.exists(separated_audio_dir):
    os.makedirs(separated_audio_dir)
    
audio_fps = sorted(glob(os.path.join(unseparated_audio_dir, "*")))
output_audio_fps = [os.path.join(separated_audio_dir, os.path.basename(x)) for x in audio_fps]

temp_df = pd.DataFrame({'input_fp' : audio_fps, 'output_audio_fps' : output_audio_fps})
temp_df_fp = os.path.join(data_dir, "temp_mixit_manifest.csv")
temp_df.to_csv(temp_df_fp)
