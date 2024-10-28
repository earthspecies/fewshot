'''
Script to process original katydid data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/katydids'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "katydid"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    if os.path.exists(dtgt_dir):
        shutil.rmtree(dtgt_dir)
    
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    
    audio_dir = DATA_DIR #os.path.join(DATA_DIR, 'audio')
        
    AD_files = ["Clip002_BCI.wav",
                "Clip006_BCI.wav",
                "Clip021_BCI.wav",
                "Clip026_BCI.wav",
               ]
    
    TS_files = ["Clip020_BCI.wav",
                "Clip024_BCI.wav",
                "Clip012_BCI.wav",
                "Clip029_BCI.wav",
               ]
    
    PT_files = ["Clip017_BCI.wav",
                "Clip028_BCI.wav",
                "Clip016_BCI.wav",
                "Clip013_BCI.wav",
               ]
        
    files_to_keep = AD_files+TS_files+PT_files
    assert len(set(files_to_keep)) == len(AD_files)+len(TS_files)+len(PT_files)
        
    for i, audio_fp in enumerate(sorted(glob(os.path.join(audio_dir, "*.wav")))):
        if os.path.basename(audio_fp) not in files_to_keep:
            continue
        
        print(f"Processing {audio_fp}")

        shutil.copy(audio_fp, audio_tgt_dir)
        manifest["audio_fp"].append(os.path.join(audio_tgt_dir, os.path.basename(audio_fp)).split("/formatted/")[1])

        # st_fp = audio_fp.replace('audio', 'selection_tables')
        st_fp = sorted(glob(audio_fp.replace('.wav', '*.txt')))[0]
        
        st = pd.read_csv(st_fp, sep='\t')
        
        st["Annotation"] = st["Analyst Tag"]
        
        if os.path.basename(audio_fp) in AD_files:
            anno="AD"
        elif os.path.basename(audio_fp) in TS_files:
            anno="TS"
        elif os.path.basename(audio_fp) in PT_files:
            anno="PT"
            
        st=st[st["Annotation"].isin([anno, "Unknown"])]

        new_st_fn = os.path.basename(st_fp)
        new_st_fp = os.path.join(annot_tgt_dir, new_st_fn)
        
        # merge overlaps
        st_annos = sorted(st["Annotation"].unique())
        
        d = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : []}
        
        for anno in st_annos:
            st_sub = st[st["Annotation"] == anno]
        
            dur_max_anno = st_sub["End Time (s)"].max() + 1
            rr = 16000
            anno_merged_np = np.zeros((int(rr * dur_max_anno),), dtype=bool)

            for i, row in st_sub.iterrows():
                begin = int(row["Begin Time (s)"] * rr)
                end = int(row["End Time (s)"] * rr)
                anno_merged_np[begin:end] = True
                
            starts = anno_merged_np[1:] * ~anno_merged_np[:-1]
            starts = np.where(starts)[0] + 1

            for start in tqdm(starts):
                look_forward = anno_merged_np[start:]
                ends = np.where(~look_forward)[0]
                if len(ends)>0:
                    end = start+np.amin(ends)
                else:
                    end = len(anno_merged_np)-1
                d["Begin Time (s)"].append(start/rr)
                d["End Time (s)"].append(end/rr)
                d["Annotation"].append(anno)

            if anno_merged_np[0]:
                start = 0
                look_forward = anno_merged_np[start:]
                ends = np.where(~look_forward)[0]
                if len(ends)>0:
                    end = start+np.amin(ends)
                else:
                    end = len(anno_merged_np)-1
                d["Begin Time (s)"].append(start/rr)
                d["End Time (s)"].append(end/rr)
                d["Annotation"].append(anno)
                
        st = pd.DataFrame(d)
        # end merge overlaps

        st.to_csv(new_st_fp, sep='\t', index=False)
        manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
