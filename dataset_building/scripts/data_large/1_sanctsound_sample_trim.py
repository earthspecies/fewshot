import soundfile as sf
import shutil
import torchaudio.transforms as T
import torchaudio
from glob import glob
import torch
from tqdm import tqdm
import os
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

TARGET_SR = 16000
MAX_FILE_DUR_SEC = 60 * 5
N_FILES_TO_SAMPLE = 10000

data_dir = '/home/jupyter/data/fewshot_data/data_large/'
gcp_addresses = os.path.join(data_dir, 'sanctsound_addresses.txt')
out_dir = os.path.join(data_dir, 'sanctsound')

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

os.makedirs(out_dir)

gcp_addresses = pd.read_csv(gcp_addresses, names=['address'])
gcp_addresses = sorted(gcp_addresses['address'].sample(N_FILES_TO_SAMPLE, random_state=0))

# Define a function to handle the download and processing for each file
def process_audio_file(address):
    try:
        dl_target = os.path.join(out_dir, address.split('/')[-1])
        subprocess.run(["gsutil", "-m", "cp", address, out_dir])

        audio, sr = torchaudio.load(dl_target)
        if len(audio.shape) > 1:
            audio = audio[0, :]

        # Trim audio to MAX_FILE_DUR_SEC
        audio = audio[:int(MAX_FILE_DUR_SEC * sr)]

        # Resample to target sampling rate
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=TARGET_SR)

        # Save processed file
        sf.write(dl_target, audio.squeeze().cpu().numpy(), TARGET_SR)
    except Exception as e:
        print(f"Failed to process {address}: {e}")

# Use ThreadPoolExecutor to parallelize the downloads and processing
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_audio_file, address) for address in gcp_addresses]

    # Use tqdm to track the progress of multithreaded tasks
    for future in tqdm(as_completed(futures), total=len(futures)):
        future.result()  # Block until each task is completed



# import soundfile as sf
# import shutil
# import torchaudio.transforms as T
# import torchaudio
# from glob import glob
# import torch
# from tqdm import tqdm
# import os
# import subprocess
# import pandas as pd

# TARGET_SR = 16000
# MAX_FILE_DUR_SEC = 60*5
# N_FILES_TO_SAMPLE = 10000

# data_dir = '/home/jupyter/data/fewshot_data/data_large/'
# gcp_addresses = os.path.join(data_dir, 'sanctsound_addresses.txt')
# out_dir = os.path.join(data_dir, 'sanctsound')

# if os.path.exists(out_dir):
#     shutil.rmtree(out_dir)
    
# os.makedirs(out_dir)

# gcp_addresses = pd.read_csv(gcp_addresses, names = ['address'])
# gcp_addresses = sorted(gcp_addresses['address'].sample(N_FILES_TO_SAMPLE, random_state=0))

# for address in tqdm(gcp_addresses):
#     dl_target = os.path.join(out_dir, address.split('/')[-1])
#     subprocess.run(["gsutil", "-m", "cp", address, out_dir])
    
#     audio, sr = torchaudio.load(dl_target)
#     if len(audio.shape) > 1:
#         audio = audio[0,:]
    
#     audio = audio[:int(MAX_FILE_DUR_SEC*sr)]
    
#     audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=TARGET_SR)
#     sf.write(dl_target, audio.squeeze().cpu().numpy(), TARGET_SR)
    