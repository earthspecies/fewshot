from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from glob import glob
import soundfile as sf
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

dataset_name = "wavcaps"
PSEUDOVOX_MANIFEST = f'/home/ubuntu/fewshot_data/data_large/{dataset_name}_pseudovox.csv'
TARGET_FP = f'/home/ubuntu/fewshot_data/data_large/{dataset_name}_pseudovox_with_birdnet.csv'

df = pd.read_csv(PSEUDOVOX_MANIFEST)

# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

def analyze_audio(row_dict):
    """Analyze a single audio file with BirdNET."""
    try:
        audio, sr = sf.read(row_dict['pseudovox_audio_fp'])
        pad_len = int(.75 * sr)
        audio_padded = np.pad(audio, (pad_len, pad_len))
        temp_fp = f'temp_{os.path.basename(row_dict["pseudovox_audio_fp"])}.wav'
        
        sf.write(temp_fp, audio_padded, sr)

        recording = Recording(
            analyzer,
            temp_fp,
            min_conf=0,  # Adjust confidence threshold as needed
        )
        recording.analyze()

        top_prob = 0
        top_sp = ""
        if len(recording.detections) > 0:
            for x in recording.detections:
                if x['confidence'] > top_prob:
                    top_prob = x['confidence']
                    top_sp = x['scientific_name']
                    
        # Clean up temporary audio file
        os.remove(temp_fp)
        
        return top_prob, top_sp
    
    except Exception as e:
        print(f"Error processing file {row_dict['pseudovox_audio_fp']}: {e}")
        return 0, ""

# Store results
results = []

# Convert each row to a dictionary for easier serialization
df_rows_as_dicts = df.to_dict(orient='records')

# Multithreading
with ProcessPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on CPU cores
    futures = [executor.submit(analyze_audio, row_dict) for row_dict in df_rows_as_dicts]
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())

# Split the results into confidence and species prediction lists
probs, top_sps = zip(*results)

# Add results to the DataFrame
df["birdnet_confidence"] = pd.Series(probs)
df["birdnet_prediction"] = pd.Series(top_sps)

# Save the DataFrame to a CSV file
df.to_csv(TARGET_FP, index=False)
