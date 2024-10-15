'''
Script to process original zebra finch data, so that training data can be used for few shot evaluation
Takes only left channel of recordings, only test data.
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/zebra_finch/test_data'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "zebra_finch"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    
    audio_dir = os.path.join(DATA_DIR, 'audio')
    annotations_dir = os.path.join(DATA_DIR, 'selection_tables')

    for i, audio_fp in enumerate(sorted(glob(os.path.join(audio_dir, "*.wav")))):
        print(f"Processing {audio_fp}")

        # shutil.copy(audio_fp, audio_tgt_dir)
        
        audio, sr = sf.read(audio_fp)
        audio = audio[:, 0] # take left channel
        
        new_audio_fp = os.path.join(audio_tgt_dir, os.path.basename(audio_fp))
        
        sf.write(new_audio_fp, audio, sr)
        
        manifest["audio_fp"].append(os.path.join(audio_tgt_dir, os.path.basename(audio_fp)).split("/formatted/")[1])

        st_fp = audio_fp.replace('audio', 'selection_tables')
        st_fp = st_fp.replace('.wav', '.Table.1.selections.txt')
        
        st = pd.read_csv(st_fp, sep='\t')

        new_st_fn = os.path.basename(st_fp)
        new_st_fp = os.path.join(annot_tgt_dir, new_st_fn)

        st.to_csv(new_st_fp, sep='\t', index=False)
        manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
