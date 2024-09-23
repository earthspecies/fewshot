'''
Script to process original humpback data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/pifsc'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    max_dur_sec = 60*60
    files_to_keep = ["Cross_A_02_060114_200928.d20.x.flac",
                     "Cross_A_02_060206_092658.d20.x.flac",
                     "Hawaii_K_06_090505_164500.d20.x.flac",
                     "Hawaii_K_16_140221_233345.df20.x.flac",
                     "Kauai_A_01_100423_084115.df20.x.flac",
                     "Ladd_S_01_070605_174615.d20.x.flac",
                     "PHR_A_01_100219_050115.df20.x.flac",
                     "Palmyra_WT_05_080613_170952.d20.x.flac",
                     "Saipan_A_05_150120_213645.df20.x.flac",
                     "Wake_S_06_160405_213345.df20.x.flac",
                    ]
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "humpback"
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
    annotations["Filename"] = annotations['flac_compressed_xwav_object'].map(lambda x : x.split('/')[-1])
    
    for i, audio_fp in enumerate(sorted(glob(os.path.join(DATA_DIR, "*.flac")))):
        if os.path.basename(audio_fp) not in files_to_keep:
            continue
        
        print(f"Processing {audio_fp}")
        
        audio, sr = sf.read(audio_fp)
        assert len(np.shape(audio)) == 1
        audio_sub = audio[:sr*max_dur_sec]
        print(f"truncated from {len(audio)/(sr*3600)} hours to {len(audio_sub)/(sr*3600)} hours")
        
        audio_tgt_fn = os.path.basename(audio_fp).replace('.flac', '_firsthour.flac')
        audio_tgt_fp = os.path.join(audio_tgt_dir, audio_tgt_fn)
        
        sf.write(audio_tgt_fp, audio_sub, sr)
        manifest["audio_fp"].append(audio_tgt_fp.split("/formatted/")[1])
        
        st = pd.DataFrame({})
        annot_sub = annotations[annotations['Filename'] == os.path.basename(audio_fp)]
        st["Begin Time (s)"] = annot_sub["subchunk_index"]*75+annot_sub["begin_rel_subchunk"]
        st["End Time (s)"] = annot_sub["subchunk_index"]*75+annot_sub["end_rel_subchunk"]
        st["Annotation"] = annot_sub["label"]
        
        st = st[~st["Annotation"].isin(["Background", "Other", "Vessel"])]
        st = st[st["End Time (s)"] < max_dur_sec]
        st = st[st["End Time (s)"] > st["Begin Time (s)"]]
        
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
        
        new_st_fn = os.path.basename(audio_tgt_fp).replace('.flac', '.txt')
        new_st_fp = os.path.join(annot_tgt_dir, new_st_fn)
        
        st.to_csv(new_st_fp, sep='\t', index=False)
        manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
