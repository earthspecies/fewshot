'''
Script to process original Ruffed Grouse data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/ruffed_grouse/ruffed_grouse_validation_set'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "ruffed_grouse"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    if os.path.exists(dtgt_dir):
        shutil.rmtree(dtgt_dir)
    
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    
    audio_dir = os.path.join(DATA_DIR, 'audio')
    # annotations_dir = os.path.join(DATA_DIR, 'selection_tables')
    
    for month in [4,5]:
        
        starttimeshift = 0
        maudio = []
        msts = []
        
        for i, audio_fp in enumerate(sorted(glob(os.path.join(audio_dir, f"**/**/*20200{month}*.wav")))):
            print(f"Processing {audio_fp}")

            potential_annotation_fp = audio_fp.replace('/audio/', '/annotations/')
            potential_annotation_fp = potential_annotation_fp.replace('.wav', '.Table.1.selections.txt')
            if not os.path.exists(potential_annotation_fp):
                continue
            st_fp = potential_annotation_fp

            audio, sr = sf.read(audio_fp)
            assert sr == 32000
            
            maudio.append(audio)
            
            st = pd.read_csv(st_fp, sep='\t')
            st["Annotation"] = st["Species"]
            st["Original Filename"] = os.path.basename(audio_fp)
            st["Begin Time (s)"] += starttimeshift
            st["End Time (s)"] += starttimeshift
            
            msts.append(st)
            
            starttimeshift += len(audio) / sr
        
        audio_tgt_fn = f"20200{month}.wav"
        audio_tgt_fp = os.path.join(audio_tgt_dir, audio_tgt_fn)
        
        new_st_fn = f"20200{month}.txt"
        new_st_fp = os.path.join(annot_tgt_dir, new_st_fn)
        
        maudio = np.concatenate(maudio)
        msts = pd.concat(msts)
        
        sf.write(audio_tgt_fp, maudio, sr)
        msts.to_csv(new_st_fp, index=False, sep='\t')
        
        manifest["audio_fp"].append(audio_tgt_fp.split("/formatted/")[1])
        manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
