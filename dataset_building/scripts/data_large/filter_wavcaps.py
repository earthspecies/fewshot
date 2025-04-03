import pandas as pd

# wavcaps_path = "/home/ubuntu/foundation-model-storage/fewshot_data/data_large/wavcaps_pseudovox_with_birdnet_with_qf_with_c.csv"
wavcaps_path = "/home/ubuntu/foundation-model-storage/fewshot_data/data_large/wavcaps_background_info.csv"

df = pd.read_csv(wavcaps_path)
print("len start:", len(df))

# Extract the filenames
filenames = df["audio_fp"].str.split("/").str[-1].str.lower()

# Get a boolean mask of rows where the filename starts with 'y'
mask = filenames.str.startswith("y")

# Get the unique file paths that will be filtered out
filtered_paths = df.loc[mask, "audio_fp"].unique()
print("len unique", )

# Now filter them out
df = df[~mask]
print("filtered len:", len(df))

# df.to_csv("/home/ubuntu/foundation-model-storage/fewshot_data/data_large/wavcaps_pseudovox_with_birdnet_with_qf_with_c_filtered.csv", index=False)
df.to_csv("/home/ubuntu/foundation-model-storage/fewshot_data/data_large/wavcaps_background_info_filtered.csv")
