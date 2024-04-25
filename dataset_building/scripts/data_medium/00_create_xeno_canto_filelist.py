import pandas as pd
import os
from tqdm import tqdm

metadata = pd.read_csv('/home/jupyter/data/fewshot_data/animalspeak_release.csv')
n_recordings_required_to_consider=25
n_to_sample_per_name = 10
n_soundscapes_to_sample = 5000
target_fp = '/home/jupyter/data/fewshot_data/data_medium/xeno_canto_filelist.txt'



metadata = metadata[metadata['source'] == "Xeno-canto"].copy().reset_index()
counts = metadata['species_common'].value_counts()
common_names_to_sample = list(counts[counts>=n_recordings_required_to_consider].index)

gs_filepaths = []

# sample focal
for i, name in tqdm(enumerate(common_names_to_sample), total = len(common_names_to_sample)):
    if name == "Soundscape":
        continue
    metadata_sub = metadata[metadata['species_common'] == name].sample(n=n_to_sample_per_name, random_state=i)
    paths = list(metadata_sub['relative_path'].map(lambda x : f"gs://animalspeak/{x}"))
    gs_filepaths.extend(paths)
    
# sample soundscapes

metadata_sub = metadata[metadata['species_common'] == "Soundscape"].sample(n=n_soundscapes_to_sample, random_state=0)
paths = list(metadata_sub['relative_path'].map(lambda x : f"gs://animalspeak/{x}"))
gs_filepaths.extend(paths)

# write

with open(target_fp, 'w') as f:
    for line in gs_filepaths:
        f.write(f"{line}\n")
