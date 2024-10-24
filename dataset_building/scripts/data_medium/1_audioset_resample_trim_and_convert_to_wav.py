import soundfile as sf
import torchaudio.transforms as T
import torchaudio
from glob import glob
import torch
from tqdm import tqdm
import os

MIXIT_SR = 22050 # BirdMixit was trained on 22050 kHz audio
MAX_FILE_DUR_SEC = 60

max_file_dur_samples = MIXIT_SR * MAX_FILE_DUR_SEC

AUDIO_FOLDER_RAW = '/home/jupyter/data/fewshot_data/data_medium/audioset_audio/'
AUDIO_FOLDER_PROCESSED = '/home/jupyter/data/fewshot_data/data_medium/audioset_audio_trimmed/'

if not os.path.exists(AUDIO_FOLDER_PROCESSED):
    os.makedirs(AUDIO_FOLDER_PROCESSED)

assumed_orig_sr=16000
resampler = T.Resample(orig_freq=assumed_orig_sr, new_freq=MIXIT_SR)

for fp in tqdm(glob(os.path.join(AUDIO_FOLDER_RAW, "*.wav"))):
    x, sr = torchaudio.load(fp)
    x = x-torch.mean(x)
    if len(x.shape) > 1:
        x = x[0,:] #torch.mean(x, 0)
        
    if x.size(0) < 100:
        continue
    
    if sr == assumed_orig_sr:
        x = resampler(x)
    else:
        x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=MIXIT_SR)
    x = x[:max_file_dur_samples]
    
    # print(x.shape)
    
    new_fn = os.path.basename(fp) #.replace('.flac', '.wav')
    new_fp = os.path.join(AUDIO_FOLDER_PROCESSED, new_fn)
    
    sf.write(new_fp, x.squeeze().cpu().numpy(), MIXIT_SR)
