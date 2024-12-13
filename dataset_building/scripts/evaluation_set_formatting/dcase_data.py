'''
Script to process original DCASE2022 or 2024 Task 5 data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil

def main():
    DEV_SET_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/DCASE_2024/Development_Set'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    
    os.makedirs(target_dir, exist_ok=True)
    
    train_dir = os.path.join(DEV_SET_DIR, "Training_Set")
    val_dir = os.path.join(DEV_SET_DIR, "Validation_Set")
    
    for dname in ["HB", "ME", "PB", "PB24", "PW", "RD"]: #["BV", "HT", "JD", "MT", "WMW", "HB", "ME", "PB", "PB24", "PW", "RD"]:
        print(f"Processing {dname}")
        dtgt_dir = os.path.join(target_dir, dname)
        audio_tgt_dir = os.path.join(dtgt_dir, "audio")
        annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
        
        os.makedirs(dtgt_dir, exist_ok=True)
        os.makedirs(audio_tgt_dir, exist_ok=True)
        os.makedirs(annot_tgt_dir, exist_ok=True)
        
        manifest = {"audio_fp": [], "selection_table_fp" : []}
        
        parent_dir = val_dir if dname in ["HB", "ME", "PB", "PB24", "PW", "RD"] else train_dir
        
        for audio_fp in sorted(glob(os.path.join(parent_dir, dname, "*.wav"))):
            shutil.copy(audio_fp, audio_tgt_dir)      
            manifest["audio_fp"].append(os.path.join(audio_tgt_dir, os.path.basename(audio_fp)).split("/formatted/")[1])
            
            annot_fp = audio_fp.replace(".wav", ".csv")
            df = pd.read_csv(annot_fp)
            annot_names = list(df.columns)
            
            st = {"Begin Time (s)":[], "End Time (s)":[], "Annotation":[]}
            
            for i, row in df.iterrows():
                for annot_name in annot_names:
                    if annot_name in ["Audiofilename", "Starttime", "Endtime", "Unknown"]:
                        continue
                    if row[annot_name] == "UNK":
                        st["Begin Time (s)"].append(row["Starttime"])
                        st["End Time (s)"].append(row["Endtime"])
                        st["Annotation"].append("Unknown")
                        break
                    elif row[annot_name] == "POS":
                        st["Begin Time (s)"].append(row["Starttime"])
                        st["End Time (s)"].append(row["Endtime"])
                        st["Annotation"].append(annot_name)
                        break
                    
            st = pd.DataFrame(st)
            st_fp = os.path.join(annot_tgt_dir, os.path.basename(annot_fp).replace(".csv", ".txt"))
            st.to_csv(st_fp, index=False, sep='\t')
            manifest["selection_table_fp"].append(st_fp.split("/formatted/")[1])
            
        pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
