import soundfile as sf
import torchaudio
from glob import glob
import torch
from tqdm import tqdm
import os

MIXIT_SR = 22050
MAX_FILE_DUR_SEC = 60

max_file_dur_samples = MIXIT_SR * MAX_FILE_DUR_SEC


AUDIO_FOLDER_RAW = '/home/jupyter/fewshot/data/data_small/audio/'
AUDIO_FOLDER_PROCESSED = '/home/jupyter/fewshot/data/data_small/audio_trimmed/'

if not os.path.exists(AUDIO_FOLDER_PROCESSED):
    os.makedirs(AUDIO_FOLDER_PROCESSED)

for fp in tqdm(glob(os.path.join(AUDIO_FOLDER_RAW, "*.flac"))):
    x, sr = torchaudio.load(fp)
    x = x-torch.mean(x)
    if len(x.shape) > 1:
        x = torch.mean(x, 0)
    
    
    x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=MIXIT_SR)
    x = x[:max_file_dur_samples]
    
    # print(x.shape)
    
    new_fn = os.path.basename(fp).replace('.flac', '.wav')
    new_fp = os.path.join(AUDIO_FOLDER_PROCESSED, new_fn)
    
    sf.write(new_fp, x.squeeze().cpu().numpy(), MIXIT_SR)