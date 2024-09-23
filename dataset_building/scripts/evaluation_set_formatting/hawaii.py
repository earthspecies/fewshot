'''
Script to process original Hawaii data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/CLO_Hawaii'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "Hawaii"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    if os.path.exists(dtgt_dir):
        shutil.rmtree(dtgt_dir)
    
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    
    annotations = pd.read_csv(os.path.join(DATA_DIR, 'annotations.csv'))
    
    # omao
    omao_files = ["UHH_535_S02_20210328_060000.flac",
                  "UHH_616_S02_20220322_074200.flac",
                  "UHH_627_S02_20220323_100400.flac",
                  "UHH_505_S02_20210327_072000.flac"
                 ]
    
    # hawpet1
    hawpet1_files = ["UHH_478_S04_20190410_210000.flac",
                     "UHH_494_S04_20190418_203000.flac",
                     "UHH_417_S03_20180522_211800.flac",
                     "UHH_439_S03_20180713_223900.flac",
                    ]
    
    # hawama
    hawama_files = ["UHH_003_S01_20161121_152000.flac",
                    "UHH_016_S01_20161122_073000.flac",
                    "UHH_077_S01_20161125_100000.flac",
                    "UHH_190_S01_20161130_102000.flac",
                   ]
    
    files_to_keep = omao_files+hawpet1_files+hawama_files
    assert len(files_to_keep) == len(set(files_to_keep))
    
    for i, audio_fp in enumerate(sorted(glob(os.path.join(DATA_DIR, "*.flac")))):
        if os.path.basename(audio_fp) not in files_to_keep:
            continue
            
        print(f"Processing {audio_fp}")
        
        shutil.copy(audio_fp, audio_tgt_dir)
        manifest["audio_fp"].append(os.path.join(audio_tgt_dir, os.path.basename(audio_fp)).split("/formatted/")[1])
        
        st = pd.DataFrame({})
        annot_sub = annotations[annotations['Filename'] == os.path.basename(audio_fp)]
        
        if os.path.basename(audio_fp) in omao_files:
            anno="omao"
        elif os.path.basename(audio_fp) in hawpet1_files:
            anno="hawpet1"
        elif os.path.basename(audio_fp) in hawama_files:
            anno="hawama"
        
        annot_sub = annot_sub[annot_sub['Species eBird Code'] == anno]
        
        st["Begin Time (s)"] = annot_sub["Start Time (s)"]
        st["End Time (s)"] = annot_sub["End Time (s)"]
        st["Annotation"] = annot_sub["Species eBird Code"]
        
        new_st_fn = os.path.basename(audio_fp).replace('.flac', '.txt')
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
