'''
Script to process original Powdermill data, so that training data can be used for few shot evaluation
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
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/Powdermill'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "Powdermill"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    if os.path.exists(dtgt_dir):
        shutil.rmtree(dtgt_dir)
    
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    
    WOTH_files = ["Recording_1", "Recording_2"]
    COYE_files = ["Recording_3", "Recording_4"]
    
    for rname in [f"Recording_{n}" for n in range(1,5)]:
        print(f"Processing {rname}")
        
        full_audio = []
        full_st = []
        
        for i, audio_fp in enumerate(sorted(glob(os.path.join(DATA_DIR, rname, "*.wav")))):
            
            audio_samples, sr = sf.read(audio_fp)
            audio_samples = audio_samples.flatten()
            assert len(audio_samples) == sr*5*60
            
            full_audio.append(audio_samples)
            
            selection_table_fp = audio_fp.replace('.wav', '.Table.1.selections.txt')
            st = pd.read_csv(selection_table_fp, sep='\t')
            
            st_reformatted = st[['Begin Time (s)', 'End Time (s)']].copy()
            st_reformatted['Annotation'] = st['Species']
            st_reformatted['Begin Time (s)'] += i*5*60
            st_reformatted['End Time (s)'] += i*5*60
            
            full_st.append(st_reformatted)
            
        full_audio = np.concatenate(full_audio)
        full_st = pd.concat(full_st).reset_index()
        
        if rname in COYE_files:
            anno = "COYE"
        elif rname in WOTH_files:
            anno = "WOTH"
            
        st=full_st[full_st["Annotation"] == anno]
        
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
        
        new_audio_fp = os.path.join(audio_tgt_dir, f"{rname}.wav")
        new_st_fp = os.path.join(annot_tgt_dir, f"{rname}.txt")
        
        sf.write(new_audio_fp, full_audio, sr)
        st.to_csv(new_st_fp, sep='\t', index=False)
        
        manifest["audio_fp"].append(new_audio_fp.split("/formatted/")[1])
        manifest["selection_table_fp"].append(new_st_fp.split("/formatted/")[1])
        
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
