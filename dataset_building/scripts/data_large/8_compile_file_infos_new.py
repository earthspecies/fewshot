import pandas as pd
import os
from glob import glob
import librosa
from tqdm import tqdm

# # Sanctsound
# print("Sanctsound")

# info = {"audio_fp" : [], "duration" : []}
# audio_fps = sorted(glob('/home/jupyter/data/fewshot_data/data_large/sanctsound/*.flac'))

# for audio_fp in tqdm(audio_fps):
#     info['audio_fp'].append(audio_fp)
#     info['duration'].append(librosa.get_duration(path=audio_fp))

# info = pd.DataFrame(info)
# info.to_csv('/home/jupyter/data/fewshot_data/data_large/sanctsound_background_info.csv', index=False)

# # DeepShip
# print("Deepship")

# info = {"audio_fp" : [], "duration" : []}
# audio_fps = sorted(glob('/home/jupyter/data/fewshot_data/data_large/deepship/**/**/*.wav'))

# for audio_fp in tqdm(audio_fps):
#     info['audio_fp'].append(audio_fp)
#     info['duration'].append(librosa.get_duration(path=audio_fp))

# info = pd.DataFrame(info)
# info.to_csv('/home/jupyter/data/fewshot_data/data_large/deepship_background_info.csv', index=False)

# # SilentCities
# print("SilentCities")

# info = {"audio_fp" : [], "duration" : []}
# audio_fps = sorted(glob('/home/jupyter/data/fewshot_data/data_large/silentcities_sampled/*.flac'))

# for audio_fp in tqdm(audio_fps):
#     info['audio_fp'].append(audio_fp)
#     info['duration'].append(librosa.get_duration(path=audio_fp))

# info = pd.DataFrame(info)
# info.to_csv('/home/jupyter/data/fewshot_data/data_large/silentcities_background_info.csv', index=False)

# # AnimalSpeak
# print("AnimalSpeak")

# info = {"audio_fp" : [], "duration" : []}
# audio_fps = sorted(glob('/home/jupyter/data/fewshot_data/data_large/animalspeak_audio_trimmed_16k/*.wav'))

# for audio_fp in tqdm(audio_fps):
#     dur = librosa.get_duration(path=audio_fp)
#     if dur < 5:
#         continue
#     info['audio_fp'].append(audio_fp)
#     info['duration'].append(dur)

# info = pd.DataFrame(info)
# info.to_csv('/home/jupyter/data/fewshot_data/data_large/animalspeak_background_info.csv', index=False)

# RIR data
# Gotten from: http://mcdermottlab.mit.edu/Reverb/IR_Survey.html
print("RIR")

info = {"audio_fp" : [], "duration" : []}
audio_fps = sorted(glob('/home/jupyter/data/fewshot_data/data_large/RIR_Audio/*.wav'))

for audio_fp in tqdm(audio_fps):
    info['audio_fp'].append(audio_fp)
    info['duration'].append(librosa.get_duration(path=audio_fp))

info = pd.DataFrame(info)
info.to_csv('/home/jupyter/data/fewshot_data/data_large/RIR_info.csv', index=False)