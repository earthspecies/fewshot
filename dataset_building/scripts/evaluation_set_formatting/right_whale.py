'''
Script to process original right whale data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/FRDR/data/continuous/dataset_B'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "right_whale"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    if os.path.exists(dtgt_dir):
        shutil.rmtree(dtgt_dir)
    
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    
    annotations = pd.read_csv(os.path.join(DATA_DIR, 'annotations_B_cont.csv'), sep=';')
    typical_call_duration_sec = 1 # based off inspection of calls. Calls are fairly stereotyped.
    
    # This is dataset B* from the publication "Performance of a deep neural network at detecting North Atlantic right whale upcalls"
    # The files here are manually annotated at midpoints
    # We choose a 1-sec window around the midpoints, to agree with the template described by this article "1-s, 100–200 Hz chirp with a 610 Hz bandwidth".
    # This is based on "North Atlantic right whale shift to the Gulf of St. Lawrence in 2015, revealed by long-term passive acoustics"

    files_to_keep = [f"B_cont_{i}.wav" for i in [2,8,12,18,22,27,33,37,42,48]]
    
    for i, audio_fp in enumerate(sorted(glob(os.path.join(DATA_DIR, "audio/*.wav")))):
        if os.path.basename(audio_fp) not in files_to_keep:
            continue
        
        print(f"Processing {audio_fp}")
        
        shutil.copy(audio_fp, audio_tgt_dir)
        manifest["audio_fp"].append(os.path.join(audio_tgt_dir, os.path.basename(audio_fp)).split("/formatted/")[1])
        
        st = pd.DataFrame({})
        annot_sub = annotations[annotations['filename'] == os.path.basename(audio_fp)]
        st["Begin Time (s)"] = annot_sub["timestamp"].map(lambda x : max(0, x- typical_call_duration_sec/2))
        st["End Time (s)"] = annot_sub["timestamp"].map(lambda x : min(30*60, x+ typical_call_duration_sec/2))
        st["Annotation"] = "POS"
        
        # merge overlaps
        st_annos = sorted(st["Annotation"].unique())
        
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

            d = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : []}

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
        
        new_st_fn = os.path.basename(audio_fp).replace('.wav', '.txt')
        new_st_fp = os.path.join(annot_tgt_dir, new_st_fn)
        
        st.to_csv(new_st_fp, sep='\t', index=False)
        manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
