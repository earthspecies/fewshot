'''
Script to process original naked mole rat data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/naked_mole_rat/data'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "naked_mole_rat"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    
    audio_dir = DATA_DIR #os.path.join(DATA_DIR, 'audio')
    # annotations_dir = os.path.join(DATA_DIR, 'selection_tables')
    sr=22050

    for i, audio_fp in enumerate(sorted(glob(os.path.join(audio_dir, "**/recordings/*.npy")))):
        print(f"Processing {audio_fp}")
        
        st_fp = audio_fp.replace('.npy', '.txt')
        if not os.path.exists(st_fp):
            continue
        
        audio = np.load(audio_fp)
        audio_tgt_fn = os.path.basename(audio_fp).replace('.npy', '.wav')
        audio_tgt_fp = os.path.join(audio_tgt_dir, audio_tgt_fn)
        
        sf.write(audio_tgt_fp, audio, sr)

        # shutil.copy(audio_fp, audio_tgt_dir)
        manifest["audio_fp"].append(audio_tgt_fp.split("/formatted/")[1])

        # st_fp = audio_fp.replace('audio', 'selection_tables')
        # st_fp = st_fp.replace('.wav', '.txt')
        
        old_st = pd.read_csv(st_fp, sep='\t')
        
        st = pd.DataFrame({})
        st["Begin Time (s)"] = old_st["s"]
        st["End Time (s)"] = old_st["e"]
        st["Annotation"] = old_st["cl"]

        new_st_fn = os.path.basename(st_fp)
        new_st_fp = os.path.join(annot_tgt_dir, new_st_fn)

        st.to_csv(new_st_fp, sep='\t', index=False)
        manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
