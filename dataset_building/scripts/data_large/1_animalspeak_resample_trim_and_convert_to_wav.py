import soundfile as sf
import torchaudio.transforms as T
import torchaudio
import torch
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import DATA_BASE

MIXIT_SR = 22050  # BirdMixit was trained on 22050 kHz audio
MAX_FILE_DUR_SEC = 60
max_file_dur_samples = MIXIT_SR * MAX_FILE_DUR_SEC

AUDIO_FOLDER_PROCESSED = f'{DATA_BASE}/fewshot_data/data_large/animalspeak_audio_trimmed/'

if not os.path.exists(AUDIO_FOLDER_PROCESSED):
    os.makedirs(AUDIO_FOLDER_PROCESSED)

assumed_orig_sr = 48000
resampler = T.Resample(orig_freq=assumed_orig_sr, new_freq=MIXIT_SR)
count_global = 0
len_global = 0

def process_file(fp):
    try:
        x, sr = torchaudio.load(fp)
        x = x - torch.mean(x)
        if len(x.shape) > 1:
            x = x[0, :]
        
        if x.size(0) < 100:
            return
        
        if sr == assumed_orig_sr:
            x = resampler(x)
        else:
            x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=MIXIT_SR)
        
        x = x[:max_file_dur_samples]
        
        new_fn = os.path.basename(fp).replace('.flac', '.wav')
        new_fp = os.path.join(AUDIO_FOLDER_PROCESSED, new_fn)
        
        sf.write(new_fp, x.squeeze().cpu().numpy(), MIXIT_SR)
        count_global += 1
        print("resampled and wrote.", count_global, len_global)
    except Exception as e:
        print(f"Error processing {fp}: {e}")

# Load file paths from a text file
input_file_list = "/home/davidrobinson/fewshot_data/data_large/xeno_canto_filelist_fixed.txt"

with open(input_file_list, 'r') as file:
    files = [line.strip() for line in file if line.strip()]
len_global = len(files)

# Use ThreadPoolExecutor for parallel processing with tqdm
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_file, fp) for fp in files]
    
    # Use tqdm to display progress
    for future in tqdm(as_completed(futures), total=len(futures)):
        # This line just ensures that exceptions are raised, allowing tqdm to update correctly
        future.result()
