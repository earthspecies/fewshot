'''
Script to process original Anuraset data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/Anuraset'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "Anuraset"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    if os.path.exists(dtgt_dir):
        shutil.rmtree(dtgt_dir)
        
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    
    audio_dir = os.path.join(DATA_DIR, 'raw_data')
    annotations_dir = os.path.join(DATA_DIR, 'strong_labels')
    
    # PHYALB
    PHYALB_files = ["INCT17_20191215_214500.wav",
                      "INCT17_20200107_180000.wav",
                      "INCT17_20200320_023000.wav",
                      "INCT17_20201112_040000.wav",
                     ]
    
    # BOALUN
    BOALUN_files = ["INCT41_20201015_203000.wav",
                      "INCT41_20201105_191500.wav",
                      "INCT41_20201121_234500.wav",
                      "INCT41_20210106_003000.wav",
                     ]
    
    # LEPLAT
    LEPLAT_files = ["INCT20955_20190911_004500.wav",
                      "INCT20955_20190911_233000.wav",
                      "INCT20955_20191210_233000.wav",
                      "INCT20955_20191229_211500.wav",
                     ]
    
    files_to_keep = PHYALB_files+BOALUN_files+LEPLAT_files
    assert len(files_to_keep) == len(set(files_to_keep))
    

    for i, audio_fp in enumerate(sorted(glob(os.path.join(audio_dir, "**/*.wav")))):
        if os.path.basename(audio_fp) not in files_to_keep:
            continue
        
        print(f"Processing {audio_fp}")

        shutil.copy(audio_fp, audio_tgt_dir)
        manifest["audio_fp"].append(os.path.join(audio_tgt_dir, os.path.basename(audio_fp)).split("/formatted/")[1])

        st_fp = audio_fp.replace('raw_data', 'strong_labels')
        st_fp = st_fp.replace('.wav', '.txt')
        
        st = pd.read_csv(st_fp, sep='\t', names=["Begin Time (s)", "End Time (s)", "Annotation"])

        new_st_fn = os.path.basename(st_fp)
        new_st_fp = os.path.join(annot_tgt_dir, new_st_fn)
        
        if os.path.basename(audio_fp) in PHYALB_files:
            anno = "PHYALB"
        elif os.path.basename(audio_fp) in BOALUN_files:
            anno = "BOALUN"
        elif os.path.basename(audio_fp) in LEPLAT_files:
            anno = "LEPLAT"
            
        st = st[st["Annotation"].map(lambda x : anno == x.split('_')[0])]
        
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

        st.to_csv(new_st_fp, sep='\t', index=False)
        manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
