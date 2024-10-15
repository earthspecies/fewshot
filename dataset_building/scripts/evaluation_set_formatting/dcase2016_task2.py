'''
Script to process original dcase 2016 task 2 data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np
import json

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/DCASE_2016_task2/hear-2021.0.6/tasks/dcase2016_task2-hear2021-full'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "dcase2016_task2"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    
    with open(os.path.join(DATA_DIR, 'test.json'), 'r') as file:
        annotations = json.load(file)
    
    for i, audio_fp in enumerate(sorted(glob(os.path.join(DATA_DIR, "48000/test/*.wav")))):
        print(f"Processing {audio_fp}")
        
        shutil.copy(audio_fp, audio_tgt_dir)
        manifest["audio_fp"].append(os.path.join(audio_tgt_dir, os.path.basename(audio_fp)).split("/formatted/")[1])
        
        begin = []
        end = []
        anno = []
        
        annot_sub = annotations[os.path.basename(audio_fp)]
        
        for k in annot_sub:
            begin.append(k['start']/1000)
            end.append(k['end']/1000)
            anno.append(k['label'])
                                
        st = pd.DataFrame({"Begin Time (s)" : begin, "End Time (s)" : end, "Annotation" : anno})
        
        new_st_fn = os.path.basename(audio_fp).replace('.wav', '.txt')
        new_st_fp = os.path.join(annot_tgt_dir, new_st_fn)
        
        st.to_csv(new_st_fp, sep='\t', index=False)
        manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
