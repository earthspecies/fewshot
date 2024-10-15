'''
Script to process original gunshots data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/gunshots/Yohetal2024_Gunshot_Sound_Dataset_and_Koogu_Model/Training_and_Test_Data'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "gunshots"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    if os.path.exists(dtgt_dir):
        shutil.rmtree(dtgt_dir)
        
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    
    annotations = pd.read_csv(os.path.join(DATA_DIR, 'test_annotations/Test_Gunshots.txt'), sep='\t')
    
    sites = sorted(glob(os.path.join(DATA_DIR, "test_audio/*.wav")))
    sites = [os.path.basename(x) for x in sites]
    sites = [x.split('_')[0] for x in sites]
    sites = sorted(set(sites))
    
    for site in sites:
        
        starttimeshift = 0
        maudio = []
        msts = []
        
        for i, audio_fp in enumerate(sorted(glob(os.path.join(DATA_DIR, f"test_audio/{site}_*.wav")))):
            print(f"Processing {audio_fp}")

            audio, sr = sf.read(audio_fp)
            assert sr == 8000
            
            maudio.append(audio)
            
            
            st = pd.DataFrame({})
            annot_sub = annotations[annotations['Begin File'] == os.path.basename(audio_fp)]
            st["Begin Time (s)"] = annot_sub["File Offset (s)"]
            st["End Time (s)"] = annot_sub["File Offset (s)"]+annot_sub["End Time (s)"]-annot_sub["Begin Time (s)"]
            st["Annotation"] = annot_sub["Tags"]
            st["Original Filename"] = os.path.basename(audio_fp)
            st["Begin Time (s)"] += starttimeshift
            st["End Time (s)"] += starttimeshift
            
            msts.append(st)
            
            starttimeshift += len(audio) / sr
        
        audio_tgt_fn = f"{site}.wav"
        audio_tgt_fp = os.path.join(audio_tgt_dir, audio_tgt_fn)
        
        new_st_fn = f"{site}.txt"
        new_st_fp = os.path.join(annot_tgt_dir, new_st_fn)
        
        maudio = np.concatenate(maudio)
        msts = pd.concat(msts)
        
        if len(msts)<7:
            continue
        
        sf.write(audio_tgt_fp, maudio, sr)
        msts.to_csv(new_st_fp, index=False, sep='\t')
        
        manifest["audio_fp"].append(audio_tgt_fp.split("/formatted/")[1])
        manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
