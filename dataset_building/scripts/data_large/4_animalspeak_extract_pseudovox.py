import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from scipy import signal
from glob import glob
from tqdm import tqdm
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed  # Use ProcessPoolExecutor for parallelism

dataset_name = "animalspeak"

from config import DATA_BASE


raw_dir = f"{DATA_BASE}/fewshot_data/data_large/{dataset_name}_audio_trimmed"
stems_dir = f"{DATA_BASE}/fewshot_data/data_large/{dataset_name}_audio_mixit"
output_dir = f"{DATA_BASE}/fewshot_data/data_large/{dataset_name}_pseudovox"
data_dir = f"{DATA_BASE}/fewshot_data/data_large"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_envelope(audio, sr, boxcar_len_samples):
    sos = signal.butter(10, 25, 'hp', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, audio)
    sq_amplitude = filtered ** 2
    envelope = np.convolve(sq_amplitude, np.ones(boxcar_len_samples) / boxcar_len_samples)
    envelope = np.sqrt(envelope)
    return envelope

def fill_holes_fast(m, max_hole):
    stops = m[:-1] * ~m[1:]
    stops = np.where(stops)[0]
    for stop in stops:
        look_forward = m[stop+1:stop+max_hole]
        if np.any(look_forward):
            next_start = np.amin(np.where(look_forward)[0]) + stop + 1
            m[stop : next_start] = True
    return m

def delete_short_fast(m, min_pos):
    starts = m[1:] * ~m[:-1]
    starts = np.where(starts)[0] + 1
    clips = []
    for start in starts:
        look_forward = m[start:]
        ends = np.where(~look_forward)[0]
        if len(ends) > 0:
            clips.append((start, start + np.amin(ends)))
    m = np.zeros_like(m).astype(bool)
    for clip in clips:
        if clip[1] - clip[0] >= min_pos:
            m[clip[0]:clip[1]] = True
    return m

def process_audio_file(raw_audio_fp):
    info = {'raw_audio_fp': [], 'stem_audio_fp': [], 'pseudovox_audio_fp': [], 'Begin Time (s)': [], 'End Time (s)': []}
    
    try:
        raw_audio, raw_sr = sf.read(raw_audio_fp)
        raw_envelope = get_envelope(raw_audio, raw_sr, raw_sr // 100)
        raw_envelope_min = np.amin(raw_envelope[raw_sr // 10:-raw_sr // 10])
        raw_envelope_max = np.amax(raw_envelope[raw_sr // 10:-raw_sr // 10])

        stem_fns = [os.path.basename(raw_audio_fp).replace(".wav", f"_source{i}.wav") for i in range(4)]
        stem_fps = [os.path.join(stems_dir, x) for x in stem_fns]

        for stem_fp in stem_fps:
            audio, sr = sf.read(stem_fp)
            envelope = get_envelope(audio, sr, raw_sr // 100)
            envelope = (envelope - raw_envelope_min) / (raw_envelope_max - raw_envelope_min + 1e-6)
            mask = envelope > .25
            filled_mask = fill_holes_fast(mask, int(sr * .1))
            thinned_mask = delete_short_fast(filled_mask, int(sr * .03))
            starts = thinned_mask[1:] * ~thinned_mask[:-1]
            starts = np.where(starts)[0] + 1
            clips = []
            for start in starts:
                look_forward = thinned_mask[start:]
                ends = np.where(~look_forward)[0]
                if len(ends) > 0:
                    clips.append((start, start + np.amin(ends)))
            for k, clip in enumerate(clips):
                s, e = clip
                pad_len = int(0.05 * sr)
                s = max(0, s - pad_len)
                e = e + pad_len
                ramp_len = int(0.05 * sr)
                on_ramp = np.linspace(0.0, 1.0, ramp_len)
                off_ramp = np.linspace(1.0, 0.0, ramp_len)
                padded_clip = audio[max(0, s - ramp_len):e + ramp_len]
                padded_clip[-ramp_len:] *= off_ramp
                padded_clip[:ramp_len] *= on_ramp
                target_fn = os.path.basename(stem_fp).replace(".wav", f"_clip{k}.wav")
                target_fp = os.path.join(output_dir, target_fn)
                sf.write(target_fp, padded_clip, sr)
                info['raw_audio_fp'].append(raw_audio_fp)
                info['stem_audio_fp'].append(stem_fp)
                info['pseudovox_audio_fp'].append(target_fp)
                info['Begin Time (s)'].append(s / sr)
                info['End Time (s)'].append(e / sr)

    except Exception as e:
        print(f"Error processing file {raw_audio_fp}: {e}")
        
    return info

raw_audio_fps = sorted(glob(os.path.join(raw_dir, "*.wav")))

info_list = []
with ProcessPoolExecutor(max_workers=16) as executor:  # Use ProcessPoolExecutor for CPU-bound tasks
    futures = {executor.submit(process_audio_file, raw_audio_fp): raw_audio_fp for raw_audio_fp in raw_audio_fps}
    for future in tqdm(as_completed(futures), total=len(futures)):
        info_list.append(future.result())

# Combine all results into a DataFrame and save as CSV
info_combined = {'raw_audio_fp': [], 'stem_audio_fp': [], 'pseudovox_audio_fp': [], 'Begin Time (s)': [], 'End Time (s)': []}

for info in info_list:
    if info:  # Ensure that info is not empty or None
        for k in info_combined.keys():
            info_combined[k].extend(info[k])

# Convert to DataFrame
info_df = pd.DataFrame(info_combined)

# Save as CSV
info_df.to_csv(os.path.join(data_dir, f"{dataset_name}_pseudovox.csv"), index=False)
