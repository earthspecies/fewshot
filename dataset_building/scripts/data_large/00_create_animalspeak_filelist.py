import pandas as pd
import os
from tqdm import tqdm
from config import DATA_BASE

metadata = pd.read_csv('/home/davidrobinson/biolingual-2/csvs/release/animalspeak2_release_16k_license_dup.csv')
n_recordings_required_to_consider = 3
n_to_sample_per_name = 50
n_soundscapes_to_sample = 20000
target_fp = f'{DATA_BASE}/fewshot_data/data_large/xeno_canto_filelist.txt'

# Filter out species with fewer than the required number of recordings
species_counts = metadata['species_common'].value_counts()
eligible_species = species_counts[species_counts >= n_recordings_required_to_consider].index
metadata = metadata[metadata['species_common'].isin(eligible_species)].copy()

# Shuffle the metadata dataframe
metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)

# Initialize counters and lists
species_count = {name: 0 for name in eligible_species}
gs_filepaths = []

# Iterate through the shuffled dataframe and select paths as needed
for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
    species = row['species_common']
    if species_count.get(species, 0) < n_to_sample_per_name and species != "Soundscape":
        species_count[species] += 1
        gs_filepaths.append(f"/home/davidrobinson/foundation-model-data/audio_48k/animalspeak/{row['local_path']}")
    elif species == "Soundscape" and len(gs_filepaths) < n_to_sample_per_name * len(eligible_species) + n_soundscapes_to_sample:
        gs_filepaths.append(f"/home/davidrobinson/foundation-model-data/audio_48k/animalspeak/{row['local_path']}")
    if all(count >= n_to_sample_per_name for name, count in species_count.items() if name != "Soundscape") and len(gs_filepaths) >= n_to_sample_per_name * len(eligible_species) + n_soundscapes_to_sample:
        break

# Write the file paths to the output file
with open(target_fp, 'w') as f:
    for line in gs_filepaths:
        f.write(f"{line}\n")
