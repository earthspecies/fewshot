import pandas as pd
import os
import soundfile as sf
from glob import glob
from tqdm import tqdm

# Step 1: Read and Prepare Non-Biological Labels
birdnet_nonbiological_labels = [
    "Engine", "Fireworks", "Gun", "Noise", "Siren",
    "Power tools", "Human vocal", "Human non-vocal", "Human whistle", ""
]

# Read and strip labels from 'nonbio.txt'
with open("/home/davidrobinson/fewshot/dataset_building/scripts/data_large/nonbio_sounds.txt") as nbf:
    other_nonbio_labels = [line.strip() for line in nbf]

# Step 2: Load and Merge Biolingual Predictions
pseudovox_info_fps = [
    "/home/davidrobinson/fewshot_data/data_large/animalspeak_pseudovox_with_birdnet.csv"
]

pseudovox_biolingual_predictions = (
    "/home/davidrobinson/fewshot_data/data_large/animalspeak_pseudovox_with_biolingual.csv"
)

# Read pseudovox_info DataFrames and concatenate
pseudovox_info_list = []
for fp in pseudovox_info_fps:
    df = pd.read_csv(fp)
    pseudovox_info_list.append(df)
pseudovox_info = pd.concat(pseudovox_info_list, ignore_index=True)

# Read biolingual predictions DataFrame
biolingual_df = pd.read_csv(pseudovox_biolingual_predictions)

# Adjust file paths in 'pseudovox_audio_fp' to match between DataFrames
pseudovox_info['pseudovox_audio_fp'] = pseudovox_info['pseudovox_audio_fp'].str.replace(
    '_pseudovox/', '_pseudovox_16k/'
)

# Adjust file paths in 'biolingual_df' to match
if 'audio_file' in biolingual_df.columns:
    biolingual_df['audio_file'] = biolingual_df['audio_file'].str.replace(
        '_pseudovox/', '_pseudovox_16k/'
    )
    # Rename 'filepath' to 'pseudovox_audio_fp' to match the other DataFrame
    biolingual_df = biolingual_df.rename(columns={'audio_file': 'pseudovox_audio_fp', 'similarity': 'biolingual_similarity'})

# Rename 'prediction' column to 'biolingual_prediction' for clarity
if 'predicted_label' in biolingual_df.columns:
    biolingual_df = biolingual_df.rename(columns={'predicted_label': 'biolingual_prediction'})

# Merge the DataFrames on 'pseudovox_audio_fp'
merged_df = pd.merge(
    pseudovox_info,
    biolingual_df,
    on='pseudovox_audio_fp',
    how='left'
)

# Step 3: Determine Biological vs. Non-Biological Labels

# Fill NaN values in predictions with empty strings
merged_df['birdnet_prediction'] = merged_df['birdnet_prediction'].fillna('')
merged_df['biolingual_prediction'] = merged_df['biolingual_prediction'].fillna('')

# Create 'fp_plus_prediction' and calculate 'duration_sec'
merged_df["fp_plus_prediction"] = merged_df["raw_audio_fp"] + merged_df["birdnet_prediction"]
merged_df["duration_sec"] = merged_df["End Time (s)"] - merged_df["Begin Time (s)"]

# Determine if the predictions are non-biological
merged_df['birdnet_is_nonbio'] = merged_df['birdnet_prediction'].isin(birdnet_nonbiological_labels)
merged_df['biolingual_is_nonbio'] = merged_df['biolingual_prediction'].isin(other_nonbio_labels)

# Determine if the predictions are biological
merged_df['birdnet_is_bio'] = ~merged_df['birdnet_is_nonbio']
merged_df['biolingual_is_bio'] = ~merged_df['biolingual_is_nonbio']

# Existing bio and nonbio DataFrames based on BirdNET only
pseudovox_info_bio = merged_df[merged_df['birdnet_is_bio']].reset_index(drop=True).copy()
pseudovox_info_nonbio = merged_df[merged_df['birdnet_is_nonbio']].reset_index(drop=True).copy()

# Save existing DataFrames
pseudovox_info_bio.to_csv(
    "/home/davidrobinson/fewshot_data/data_large/pseudovox_bio.csv", index=False
)
pseudovox_info_nonbio.to_csv(
    "/home/davidrobinson/fewshot_data/data_large/pseudovox_nonbio.csv", index=False
)

# Create conditions for both predictions being bio or nonbio
condition_bio_both = merged_df['birdnet_is_bio'] & merged_df['biolingual_is_bio']
condition_nonbio_both = merged_df['birdnet_is_nonbio'] & merged_df['biolingual_is_nonbio']

# Step 4: Save New DataFrames Accounting for Both Predictions

# DataFrame where both predictions are biological
merged_df_bio_both = merged_df[condition_bio_both].reset_index(drop=True)

# DataFrame where both predictions are non-biological
merged_df_nonbio_both = merged_df[condition_nonbio_both].reset_index(drop=True)

# Save the new DataFrames to CSV files
merged_df_bio_both.to_csv(
    "/home/davidrobinson/fewshot_data/data_large/pseudovox_bio_both.csv", index=False
)
merged_df_nonbio_both.to_csv(
    "/home/davidrobinson/fewshot_data/data_large/pseudovox_nonbio_both.csv", index=False
)

# For xeno-canto, TUT, and audioset, make a background audio CSV.
# sources = ["TUT", "xeno_canto", "audioset"]
sources = ["animalspeak"]

for source in sources:
    AUDIO_FOLDER_PROCESSED = f'/home/davidrobinson/fewshot_data/data_large/{source}_audio_trimmed_16k/'
    TARGET_FP = f'/home/davidrobinson/fewshot_data/data_large/{source}_background_audio_info.csv'

    audio_files = sorted(glob(os.path.join(AUDIO_FOLDER_PROCESSED, '*.wav')))
    filtered_files = []
    durs = []

    for fp in tqdm(audio_files):
        try:
            d = sf.info(fp).duration
            durs.append(d)
            filtered_files.append(fp)
        except Exception as e:
            print(e)

    df = pd.DataFrame({'raw_audio_fp': filtered_files, 'duration_sec': durs})
    df.to_csv(TARGET_FP, index=False)
