import pandas as pd
import os
import soundfile as sf
from glob import glob
from tqdm import tqdm

birdnet_nonbiological_labels = ["Engine", "Fireworks", "Gun", "Noise", "Siren", "Power tools", "Human vocal", "Human non-vocal", "Human whistle", ""]
with open("nonbio.txt") as nbf:
    other_nonbio_labels = nbf.readlines()

# Get pseudovoxes from biological sources together in one csv (birdnet confidence >0).
# Get pseudovoxes from non-biological sources together in another csv. NaN to "".
# For each, make a fine cluster column for birdnet label + original filepath

pseudovox_info_fps = [f"/home/davidrobinson/fewshot_data/data_large/animalspeak_pseudovox_with_birdnet.csv" for i in range(1)]
# pseudovox_info_fps.extend(["/home/davidrobinson/fewshot_data/data_large/audioset_pseudovox_with_birdnet.csv",
#                            "/home/davidrobinson/fewshot_data/data_large/TUT_pseudovox_with_birdnet.csv",]
#                          )

pseudovox_biolingual_predictions = f"/home/davidrobinson/fewshot_data/data_large/animalspeak_pseudovox_with_biolingual.csv"

pseudovox_info = []
for fp in pseudovox_info_fps:
    df = pd.read_csv(fp)
    pseudovox_info.append(df)
pseudovox_info = pd.concat(pseudovox_info)

# join biolingual preds
biolingual_df = pd.read_csv(pseudovox_biolingual_predictions)

# biolinugal_df[] = pseudovox_info["pseudovox_audio_fp"].str.replace("_pseudovox/", "_pseudovox_16k/")


pseudovox_info['birdnet_prediction'] = pseudovox_info['birdnet_prediction'].fillna("")
pseudovox_info["fp_plus_prediction"] = pseudovox_info["raw_audio_fp"] + pseudovox_info["birdnet_prediction"]
pseudovox_info["duration_sec"] = pseudovox_info["End Time (s)"] - pseudovox_info["Begin Time (s)"]
pseudovox_info["pseudovox_audio_fp"] = pseudovox_info["pseudovox_audio_fp"].str.replace("_pseudovox/", "_pseudovox_16k/")

pseudovox_info_bio = pseudovox_info[~pseudovox_info['birdnet_prediction'].isin(birdnet_nonbiological_labels)].reset_index().copy()
pseudovox_info_nonbio = pseudovox_info[pseudovox_info['birdnet_prediction'].isin(birdnet_nonbiological_labels)].reset_index().copy()

pseudovox_info_bio.to_csv("/home/davidrobinson/fewshot_data/data_large/pseudovox_bio.csv", index=False)
pseudovox_info_nonbio.to_csv("/home/davidrobinson/fewshot_data/data_large/pseudovox_nonbio.csv", index=False)

# For xeno-canto, TUT, and audioset, make a background audio csv.

sources = ["TUT", "xeno_canto", "audioset"]

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


    df = pd.DataFrame({'raw_audio_fp' : filtered_files, 'duration_sec' : durs})

    df.to_csv(TARGET_FP, index=False)
