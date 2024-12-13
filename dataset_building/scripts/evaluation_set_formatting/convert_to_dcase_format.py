import pandas as pd
import os
from glob import glob
import shutil
import soundfile as sf
from tqdm import tqdm
import subprocess

source_data_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted"
target_data_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted_for_dcase"

datasets = ["marmoset", "Anuraset", "carrion_crow", "katydid", "Spiders", "rana_sierrae", "Powdermill",  "Hawaii", "right_whale", "gibbons", "gunshots", "humpback", "ruffed_grouse", "audioset_strong", "DESED"]

if not os.path.exists(target_data_dir):
    os.makedirs(target_data_dir)
    
for dataset in tqdm(datasets):
    s = os.path.join(source_data_dir, dataset)
    t = os.path.join(target_data_dir, dataset)
    if not os.path.exists(t):
        os.makedirs(t)
    
    manifest = pd.read_csv(os.path.join(s, "manifest.csv"))
    
    for i, row in manifest.iterrows():
        shutil.copy(os.path.join(source_data_dir, row["audio_fp"]), t)
        audiofn = '.'.join(os.path.basename(row["audio_fp"]).split('.')[:-1])+".wav"
        
        if row['audio_fp'][-3:] != 'wav':
            subprocess.run(["ffmpeg", "-i", os.path.join(t, os.path.basename(row['audio_fp'])), os.path.join(t, audiofn)])
        
        st = pd.read_csv(os.path.join(source_data_dir, row["selection_table_fp"]), sep='\t')
        
        dcase_st = pd.DataFrame({})
        
        dcase_st["Starttime"] = st["Begin Time (s)"]
        dcase_st["Endtime"] = st["End Time (s)"]
        
        def convert_anno(x):
            if x == "Unknown":
                return "UNK"
            elif x == "UNK":
                return "UNK"
            else:
                return "POS"
        
        print(st["Annotation"].unique())
        dcase_st["Q"] = st["Annotation"].map(convert_anno)
        
        dcase_st["Audiofilename"] = audiofn
        
        dcase_st.to_csv(os.path.join(t, audiofn.replace(".wav", ".csv")), index=False)
    
