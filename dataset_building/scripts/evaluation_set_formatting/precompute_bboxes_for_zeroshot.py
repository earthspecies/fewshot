import pandas as pd
import pypandoc
import yaml
from glob import glob
import os
import numpy as np
from tqdm import tqdm

base_dir = '/home/jupyter/data/fewshot_data/evaluation/bboxes/'

datasets = ["marmoset", "Anuraset", "carrion_crow", "katydid", "Spiders", "rana_sierrae", "Powdermill",  "Hawaii", "right_whale", "gibbons", "gunshots", "humpback", "ruffed_grouse", "HB", "PB", "PB24", "RD", "PW", "ME"]

results = {}

for dataset in datasets:
    st_fps = sorted(glob(os.path.join(base_dir, dataset, 'selection_tables/*_AC.txt')))
    print(f"For {dataset}, considering {len(st_fps)} files")
    
    if dataset in ["marmoset", "Spiders", "PB", "PB24"]:
        slow_dataset = f"{dataset}_halftime"
        
    elif dataset in ["katydid"]:
        slow_dataset = f"{dataset}_sixthtime"
        
    else:
        slow_dataset = None
    
    for st_fp in st_fps:
        st = pd.read_csv(st_fp, sep='\t')
        st["Duration"] = st["End Time (s)"] - st["Begin Time (s)"]
        
        if "Annotation" in st.columns:
            st = st[st['Annotation'] != "Unknown"].sort_values("End Time (s)").reset_index().copy()
        else:
            print(f"No annotation column in {st_fp}")
            st = st.sort_values("End Time (s)").reset_index().copy()
        
        breakpoint = sorted(st["End Time (s)"].unique())[4]
        
        st_sub = st[st["End Time (s)"] <= breakpoint]
        
        dur_med = float(st_sub["Duration"].median())
        high_freq_med = float(st_sub["High Freq (Hz)"].median())
        low_freq_med = float(st_sub["Low Freq (Hz)"].median())
        
        # orig version
        orig_anno_fn = dataset + "_" + os.path.basename(st_fp).replace("_AC.txt", ".csv")
        if dataset in ["Spiders"]:
            orig_anno_fn = orig_anno_fn.split('.')[0] + ".csv"
        if dataset in ["carrion_crow"]:
            orig_anno_fn = orig_anno_fn.replace(".csv", "truncated.csv")
        if dataset in ["gibbons"]:
            orig_anno_fn = orig_anno_fn.replace("_test", "")
        if dataset in ["katydid"]:
            orig_anno_fn = orig_anno_fn.split("BCI")[0] + "BCI.csv"
        
        results[orig_anno_fn] = {"Duration" : dur_med, "High Freq (Hz)" : high_freq_med, "Low Freq (Hz)" : low_freq_med}
        
        if dataset in ["marmoset", "Spiders", "PB", "PB24"]:
            orig_anno_fn = slow_dataset + "_" + os.path.basename(st_fp).replace("_AC.txt", ".csv")
            if dataset in ["Spiders"]:
                orig_anno_fn = orig_anno_fn.split('.')[0] + ".csv"  
            if dataset in ["carrion_crow"]:
                orig_anno_fn = orig_anno_fn.replace(".csv", "truncated.csv")
            if dataset in ["gibbons"]:
                orig_anno_fn = orig_anno_fn.replace("_test", "")
                
            results[orig_anno_fn] = {"Duration" : dur_med*2, "High Freq (Hz)" : high_freq_med/2, "Low Freq (Hz)" : low_freq_med/2}

        if dataset in ["katydid"]:
            orig_anno_fn = slow_dataset + "_" + os.path.basename(st_fp).replace("_AC.txt", ".csv")
            if dataset in ["katydid"]:
                orig_anno_fn = orig_anno_fn.split("BCI")[0] + "BCI.csv"
            results[orig_anno_fn] = {"Duration" : dur_med*6, "High Freq (Hz)" : high_freq_med/6, "Low Freq (Hz)" : low_freq_med/6}

with open(os.path.join(base_dir, "precomputed.yaml"), 'w') as f:
    yaml.dump(results, f)
    