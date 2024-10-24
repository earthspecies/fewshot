import pandas as pd
import shutil
import os
from tqdm import tqdm

tqdm.pandas()

df = pd.read_csv('/home/jupyter/data/fewshot_data/data_large/animalspeak_pseudovox_with_birdnet.csv')
df['pseudovox_audio_fn'] = df['pseudovox_audio_fp'].map(lambda x : os.path.basename(x))
# df['pseudovox_audio_fn_noext'] = df['pseudovox_audio_fp'].map(lambda x : os.path.splitext(os.path.basename(x))[0])
df['raw_audio_fn_noext'] = df['raw_audio_fp'].map(lambda x : os.path.splitext(os.path.basename(x))[0])
df['on_computer'] = df['pseudovox_audio_fn'].map(lambda x : os.path.join('/home/jupyter/data/fewshot_data/data_large/animalspeak_pseudovox/', x))
df["exclude"] = ""

metadata = pd.read_csv('/home/jupyter/data/fewshot_data/animalspeak2_release_16khz.csv')
metadata['filename'] = metadata['local_path'].map(lambda x : os.path.splitext(os.path.basename(x))[0])

fn_to_source = {}

for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
    fn = row['filename']
    fn_to_source[fn] = row['source']
    
df['source'] = df['raw_audio_fn_noext'].progress_map(lambda x: fn_to_source[x])

data_dir = '/home/jupyter/data/fewshot_data/quality_filter'

if os.path.exists(data_dir):
    shutil.rmtree(data_dir)

splits = ["Train", "Val", "Test"]
split_dfs = {x : [] for x in splits}
n_per_split = {"Train" : 200, "Val" : 20, "Test" : 20}
n_groups = len(df['source'].unique())

for group in sorted(df['source'].unique()):
    total_to_sample = sum([n_per_split[x] for x in splits])
    sampled_files = df[df['source'] == group].sample(total_to_sample, random_state = 0).reset_index(drop=True)
    startidx = 0
    for split in splits:
        endidx = startidx + n_per_split[split]
        sampled_split = sampled_files.iloc[startidx:endidx]

        target_dir = os.path.join(data_dir, split)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        split_dfs[split].append(sampled_split) #
        for i, row in sampled_split.iterrows():
            shutil.copy(row['on_computer'], target_dir)

        startidx = endidx
        
for split in splits:
    sdf = split_dfs[split]
    sdf = pd.concat(sdf)
    sdf = sdf.sort_values('pseudovox_audio_fn')
    sdf.to_csv(os.path.join(data_dir, f"{split}_info.csv"), index=False)
