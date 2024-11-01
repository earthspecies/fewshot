from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from glob import glob
import soundfile as sf
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle  # Import pickle to save the embeddings dictionary

dataset_name = "animalspeak"
PSEUDOVOX_MANIFEST = f'/home/davidrobinson/fewshot_data/data_large/{dataset_name}_pseudovox.csv'
TARGET_FP = f'/home/davidrobinson/fewshot_data/data_large/{dataset_name}_pseudovox_with_birdnet_wembeddings.csv'
EMBEDDINGS = True

df = pd.read_csv(PSEUDOVOX_MANIFEST)
analyzer = Analyzer()
# Load and initialize the BirdNET-Analyzer models.

def analyze_audio(row_dict):
    """Analyze a single audio file with BirdNET."""
    try:        
        audio_fp = row_dict['pseudovox_audio_fp']
        audio, sr = sf.read(audio_fp)
        pad_len = int(.75 * sr)
        audio_padded = np.pad(audio, (pad_len, pad_len))
        temp_fp = f'temp_{os.path.basename(audio_fp)}.wav'
        
        sf.write(temp_fp, audio_padded, sr)

        recording = Recording(
            analyzer,
            temp_fp,
            min_conf=0,  # Adjust confidence threshold as needed
        )

        if EMBEDDINGS:
            recording.extract_embeddings()
        else:
            recording.analyze()
            # Access embeddings as recording.embeddings

        top_prob = 0
        top_sp = ""
        if not EMBEDDINGS:
            if len(recording.detections) > 0:
                for x in recording.detections:
                    if x['confidence'] > top_prob:
                        top_prob = x['confidence']
                        top_sp = x['scientific_name']
        
                    
        embeddings = recording.embeddings if EMBEDDINGS else None  # Get the embeddings
        
        # Clean up temporary audio file
        os.remove(temp_fp)
        
        return audio_fp, top_prob, top_sp, embeddings  # Return embeddings
    
    except Exception as e:
        print(f"Error processing file {audio_fp}: {e}")
        return audio_fp, 0, "", None  # Return None for embeddings on error

# Initialize lists to store results
file_paths = []
probs = []
top_sps = []
embeddings_list = []

# Convert each row to a dictionary for easier serialization
df_rows_as_dicts = df.to_dict(orient='records')
file_paths = []
probs = []
top_sps = []
embeddings_list = []

# Multithreading
with ProcessPoolExecutor(max_workers=32) as executor:  # Adjust max_workers based on CPU cores
    futures = [executor.submit(analyze_audio, row_dict) for row_dict in df_rows_as_dicts]
    for future in tqdm(as_completed(futures), total=len(futures)):
        audio_fp, top_prob, top_sp, embeddings = future.result()
        file_paths.append(audio_fp)
        probs.append(top_prob)
        top_sps.append(top_sp)
        embeddings_list.append(embeddings)

        if len(embeddings_list) % 100000 == 0:

                    # Save the DataFrame to a CSV file
            df.to_csv(TARGET_FP, index=False)

            # Create a dictionary mapping from file paths to embeddings
            embedding_dict = {fp: emb for fp, emb in zip(file_paths, embeddings_list) if emb is not None}

            # Save the embeddings dictionary to a pickle file
            if EMBEDDINGS:
                with open('embeddings.pkl', 'wb') as f:
                    pickle.dump(embedding_dict, f)


# Add results to the DataFrame
df["birdnet_confidence"] = pd.Series(probs)
df["birdnet_prediction"] = pd.Series(top_sps)

# Save the DataFrame to a CSV file
df.to_csv(TARGET_FP, index=False)

# Create a dictionary mapping from file paths to embeddings
embedding_dict = {fp: emb for fp, emb in zip(file_paths, embeddings_list) if emb is not None}

# Save the embeddings dictionary to a pickle file
if EMBEDDINGS:
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embedding_dict, f)
