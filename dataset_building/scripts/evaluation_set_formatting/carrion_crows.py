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
from tqdm import tqdm

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/carrion_crow'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "carrion_crow"
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
    annotations_dir = os.path.join(DATA_DIR, 'selection_tables')
    
    label_mapping = {"?" : "Unknown",
                     "crow_undeter" : "Unknown",
                     "crowchicks" : "crowchicks",
                     "cuckoo" : "cuckoo",
                     "focal" : "adult_crow",
                     "focal?" : "adult_crow",
                     "nest" : "Unknown",
                     "not focal" : "adult_crow",
                     "not focal LD" : "Unknown",
                     "not focal?" : "adult_crow",
                    }
    
    adult_crow_files = ["19_BA_Amarilla_1010.wav",
                        "19_BD_Rosa_1004.wav",
                        "19_BUCK_Azul_1011.wav",
                        "19_BV_Morado_1012.wav",
                        "19_CW_Amarillo_1019.wav",
                       ]
    
    cuckoo_files = ["19_AL_Azul_1008.wav",
                    "19_AL_Azul_1015.wav",
                    "19_AL_Naranja_1006.wav",
                    "19_AL_Naranja_1014.wav",
                    "19_N7_Morado_1005.wav",
                   ]
    
    files_to_keep = adult_crow_files + cuckoo_files
    assert len(set(files_to_keep)) == len(adult_crow_files) + len(cuckoo_files)
    
    max_dur_sec = 60*60

    for i, audio_fp in enumerate(sorted(glob(os.path.join(audio_dir, "*.wav")))):
        if os.path.basename(audio_fp) not in files_to_keep:
            continue
            
        print(f"Processing {audio_fp}")
        
        audio, sr = sf.read(audio_fp)
        assert len(np.shape(audio)) == 1
        audio_sub = audio[:sr*max_dur_sec]
        print(f"truncated from {len(audio)/(sr*3600)} hours to {len(audio_sub)/(sr*3600)} hours")
        
        audio_tgt_fp = os.path.join(audio_tgt_dir, os.path.basename(audio_fp).replace(".wav", "truncated.wav"))
        sf.write(audio_tgt_fp, audio_sub, sr)
        
        manifest["audio_fp"].append(audio_tgt_fp.split("/formatted/")[1])

        st_fp = audio_fp.replace('audio', 'selection_tables')
        st_fp = st_fp.replace('.wav', '.txt')
        
        st = pd.read_csv(st_fp, sep='\t')
        st["Annotation"] = st["Annotation"].map(lambda x : label_mapping[x])
        
        st = st[st["End Time (s)"] <= max_dur_sec]
        
        if os.path.basename(audio_fp) in adult_crow_files:
            anno="adult_crow"
        elif os.path.basename(audio_fp) in cuckoo_files:
            anno="cuckoo"
            
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
