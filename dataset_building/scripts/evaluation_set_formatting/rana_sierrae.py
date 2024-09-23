'''
Script to process original Rana sierrae data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/Rana_sierrae/rana_sierrae_2022'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "rana_sierrae"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    if os.path.exists(dtgt_dir):
        shutil.rmtree(dtgt_dir)
        
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    
    audio_dir = os.path.join(DATA_DIR, 'mp3')
    
    type_E_days = [f"202206{x}" for x in [21,23,25]]
    type_A_days = [f"202206{x}" for x in [20,22,24,26]]
    
    for day in [f"202206{x}" for x in range(20,27)]:
        
        full_audio = []
        full_st = []
        sr=None
        shift = 0
        
        for i, audio_fp in enumerate(sorted(glob(os.path.join(audio_dir, f"*{day}*.mp3")))):
            print(f"Processing {audio_fp}")

            audio, new_sr = sf.read(audio_fp)
            
            if sr != None:
                assert new_sr == sr
            sr=new_sr
                
            full_audio.append(audio)

            st_fp = audio_fp.replace('/mp3/', '/raven_selection_tables/')
            st_fp = st_fp.replace('.mp3', '.Table.1.selections.txt')

            st = pd.read_csv(st_fp, sep='\t')
            st["Annotation"] = st["annotation"]
            name_conversion_dict = {"A" : "primary_vocalization",
                                    "B" : "stuttered_vocalization",
                                    "C" : "chuck",
                                    "D" : "short_downward_single_note",
                                    "E" : "frequency_modulated_call",
                                    "X" : "Unknown"
                                   }

            st["Annotation"] = st["Annotation"].map(lambda x : name_conversion_dict[x])
            
            st['Begin Time (s)'] += shift
            st['End Time (s)'] += shift
            
            full_st.append(st)
            
            shift+=len(audio)/sr
            
            
        audio = np.concatenate(full_audio)
        new_audio_fn = f"{day}.mp3"
        sf.write(os.path.join(audio_tgt_dir, new_audio_fn), audio, sr)
        
        manifest["audio_fp"].append(os.path.join(audio_tgt_dir, new_audio_fn).split("/formatted/")[1])
        
        st = pd.concat(full_st)
        
        if day in type_E_days:
            anno = "frequency_modulated_call"
        elif day in type_A_days:
            anno = "primary_vocalization"
            
        st = st[st["Annotation"].isin([anno, "Unknown"])]
        
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

            for start in starts:
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

        new_st_fn = f"{day}.txt"
        new_st_fp = os.path.join(annot_tgt_dir, new_st_fn)

        st.to_csv(new_st_fp, sep='\t', index=False)
        manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
