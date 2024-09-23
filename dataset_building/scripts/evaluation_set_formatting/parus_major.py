'''
Script to process original Crow data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/parus_major'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "parus_major"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    
    audio_dir = DATA_DIR
    # annotations_dir = os.path.join(DATA_DIR, 'selection_tables')

    for i, audio_fp in enumerate(sorted(glob(os.path.join(audio_dir, "**/*.wav")))):
        print(f"Processing {audio_fp}")
        
        potential_annotation_fp = sorted(glob(audio_fp.replace('.wav', '*.txt')))
        if len(potential_annotation_fp) == 0:
            continue
        st_fp = potential_annotation_fp[0]
        
        if "first_hr" in st_fp:
            continue
        elif "2hrs" in st_fp:
            continue
        elif "3HR" in st_fp:
            continue
        
        shutil.copy(audio_fp, audio_tgt_dir)
        manifest["audio_fp"].append(os.path.join(audio_tgt_dir, os.path.basename(audio_fp)).split("/formatted/")[1])
        
        st = pd.read_csv(st_fp, sep='\t')
        st["Annotation"] = "POS"

        new_st_fn = os.path.basename(st_fp)
        new_st_fp = os.path.join(annot_tgt_dir, new_st_fn)

        st.to_csv(new_st_fp, sep='\t', index=False)
        manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
