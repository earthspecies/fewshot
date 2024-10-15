'''
Script to process original Marmoset data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/marmoset/InfantMarmosetsVox'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "marmoset"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    if os.path.exists(dtgt_dir):
        shutil.rmtree(dtgt_dir)
    
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    
    annotations = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))
    annotations = annotations[~annotations["calltype"].isin([11,12])].copy().reset_index()
    
    callnames = {0 : "Unknown", #"Peep(Pre-Phee)",
                 1 : "Phee",
                 2 : "Twitter",
                 3 : "Trill",
                 4 : "Unknown", #"Trillphee",
                 5 : "TsikTse",
                 6 : "Egg",
                 7 : "Unknown", #"Pheecry(cry)",
                 8 : "Unknown", #"TrllTwitter",
                 9 : "Unknown", #"Pheetwitter",
                 10 : "Peep", 
                 #11 : "Silence",
                 #12 : "Noise",
                }
    
    phee_files = ["20160907_Twin1_marmoset1.wav",
                  "20160907_Twin2_marmoset1.wav",
                  "20160923_Twin3_marmoset1.wav",
                  "20160920_Twin4_marmoset1.wav",
                  "20160917_Twin5_marmoset1.wav",
                 ]
    twitter_files = ["20160907_Twin1_marmoset2.wav",
                     "20160907_Twin2_marmoset2.wav",
                     "20160923_Twin3_marmoset2.wav",
                     "20160920_Twin4_marmoset2.wav",
                     "20160917_Twin5_marmoset2.wav",
                    ]
    
    files_to_keep = phee_files+twitter_files
    assert len(set(files_to_keep))==len(phee_files)+len(twitter_files)
    
    for i, audio_fp in enumerate(sorted(glob(os.path.join(DATA_DIR, "data/**/*.wav")))):
        if os.path.basename(audio_fp) not in files_to_keep:
            continue
        
        print(f"Processing {audio_fp}")
        
        shutil.copy(audio_fp, audio_tgt_dir)
        manifest["audio_fp"].append(os.path.join(audio_tgt_dir, os.path.basename(audio_fp)).split("/formatted/")[1])
        
        st = pd.DataFrame({})
        annot_sub = annotations[annotations['filename'] == os.path.basename(audio_fp).split('.')[0]]
        st["Begin Time (s)"] = annot_sub["start"]
        st["End Time (s)"] = annot_sub["end"]
        st["Annotation"] = annot_sub["calltype"].map(lambda x : callnames[x])
        
        if os.path.basename(audio_fp) in phee_files:
            anno="Phee"
        elif os.path.basename(audio_fp) in twitter_files:
            anno="Trill"
            
        st = st[st["Annotation"].isin([anno, "Unknown"])]
        
        new_st_fn = os.path.basename(audio_fp).replace('.wav', '.txt')
        new_st_fp = os.path.join(annot_tgt_dir, new_st_fn)
        
        st.to_csv(new_st_fp, sep='\t', index=False)
        manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
