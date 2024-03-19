'''
Script to process original DCASE2022 Task 5 data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil

def main():
    DEV_SET_DIR='/home/jupyter/fewshot/data/DCASE2022_Development_Set'
    
    target_dir = os.path.join(DEV_SET_DIR, "Development_Set")
    os.makedirs(target_dir, exist_ok=True)
    
    train_dir = os.path.join(DEV_SET_DIR, "Training_Set")
    val_dir = os.path.join(DEV_SET_DIR, "Validation_Set")
    
    # copy val files
    for dname in ["HB", "ME", "PB"]:
        src = os.path.join(val_dir, dname)
        tgt = os.path.join(target_dir, dname)
        if not os.path.exists(tgt):
            shutil.copytree(src, tgt)
        
    # edit HB annotations
    for fn in ["file_423_487.csv", "file_97_113.csv"]:
        fp = os.path.join(target_dir, "HB", fn)
        df = pd.read_csv(fp)
        df["Audiofilename"] = fn.replace(".csv", ".wav")
        df.to_csv(fp, index=False)
        
    for dname in ["BV", "HT", "JD", "MT", "WMW"]:
        dtgt_dir = os.path.join(target_dir, dname)
        os.makedirs(dtgt_dir, exist_ok=True)
        for audio_fp in sorted(glob(os.path.join(train_dir, dname, "*.wav"))):
            annot_fp = audio_fp.replace(".wav", ".csv")
            df = pd.read_csv(annot_fp)
            
            annot_names = list(df.columns)
            
            for annot_name in annot_names:
                if annot_name in ["Audiofilename", "Starttime", "Endtime"]:
                    continue
                df_sub = df[df[annot_name].isin(["POS", "UNK"])]
                df_pos = df[df[annot_name].isin(["POS"])]
                if len(df_pos)<=6:
                    continue #skip cases where too few pos labels
                    
                df_sub["Q"] = df_sub[annot_name]
                df_sub["Audiofilename"] = os.path.basename(audio_fp).replace(".wav", f"_{annot_name}.wav")
                annot_tgt_fn = os.path.basename(annot_fp).replace(".csv", f"_{annot_name}.csv")
                audio_tgt_fn = os.path.basename(audio_fp).replace(".wav", f"_{annot_name}.wav")
                annot_tgt_fp = os.path.join(dtgt_dir, annot_tgt_fn)
                audio_tgt_fp = os.path.join(dtgt_dir, audio_tgt_fn)
                df_sub.to_csv(annot_tgt_fp, index=False)
                shutil.copy(audio_fp, audio_tgt_fp)
                
        
        

if __name__ == "__main__":
    main()
