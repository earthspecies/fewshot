import os
import pandas as pd
import librosa
import soundfile as sf
import shutil
import numpy as np

dataset_names = ["carrion_crow"] #["marmoset", "carrion_crow", "katydid", "Spiders", "rana_sierrae", "Powdermill", "Anuraset", "Hawaii", "right_whale", "gibbons", "gunshots", "humpback", "ruffed_grouse"]
parent_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted"


def get_breakpoint(st, break_after = 5):
    st_sub = st[st["Annotation"] != "Unknown"]
    endtime = sorted(st_sub["End Time (s)"])[break_after-1]
    return endtime

for dataset in dataset_names:
    print(f"Processing {dataset}")
    dataset_dir = os.path.join(parent_dir, dataset)
    new_dataset_dir = os.path.join(parent_dir, dataset+"_crossfile")
    
    if os.path.exists(new_dataset_dir):
        shutil.rmtree(new_dataset_dir)
    
    new_audio_dir = os.path.join(new_dataset_dir, "audio")
    new_st_dir = os.path.join(new_dataset_dir, "selection_tables")
    
    for d in [new_audio_dir, new_st_dir]:
        os.makedirs(d)
    
    manifest_fp = os.path.join(dataset_dir, "manifest.csv")
    manifest=pd.read_csv(manifest_fp)
    new_manifest = {"audio_fp" : [], "selection_table_fp" : []}
    
    # Get annot groups
    label_to_files = {}
    for i, row in manifest.iterrows():
        st = pd.read_csv(os.path.join(parent_dir, row["selection_table_fp"]), sep='\t')
        st = st[st["Annotation"] != "Unknown"]
        annot = list(st["Annotation"].unique())[0]
        if annot not in label_to_files:
            label_to_files[annot] = {"audio_fp" : [], "selection_table_fp" : []}
        label_to_files[annot]["audio_fp"].append(row["audio_fp"])
        label_to_files[annot]["selection_table_fp"].append(row["selection_table_fp"])
        
    for annot in label_to_files:
        sub_manifest = pd.DataFrame(label_to_files[annot])
        sub_manifest = sub_manifest.sample(frac=1, random_state=907).reset_index()
        print(f"Processing {dataset} with annotation {annot}, there are {len(sub_manifest)} files")
        
        audio_fp_shifted = list(sub_manifest["audio_fp"][1:]) + list(sub_manifest["audio_fp"][:1])
        st_fp_shifted = list(sub_manifest["selection_table_fp"][1:]) + list(sub_manifest["selection_table_fp"][:1])
        
        sub_manifest["audio_fp_shifted"] = audio_fp_shifted
        sub_manifest["selection_table_fp_shifted"] = st_fp_shifted
        
        for i, row in sub_manifest.iterrows():
            support_st = pd.read_csv(os.path.join(parent_dir, row["selection_table_fp"]), sep = "\t")
            query_st = pd.read_csv(os.path.join(parent_dir, row["selection_table_fp_shifted"]), sep = "\t")
            
            support_breakpoint = get_breakpoint(support_st)
            query_breakpoint = get_breakpoint(query_st)
            
            support_audio, sr = librosa.load(os.path.join(parent_dir, row["audio_fp"]), duration = support_breakpoint, sr=None)
            query_audio, sr = librosa.load(os.path.join(parent_dir, row["audio_fp_shifted"]), offset = query_breakpoint, sr=sr)
            
            new_audio = np.concatenate([support_audio, query_audio], axis=-1)
            
            support_st_sub = support_st[support_st["Begin Time (s)"] < support_breakpoint].copy()
            query_st_sub = query_st[query_st["End Time (s)"] > query_breakpoint].copy()
            query_st_sub["Begin Time (s)"] = query_st_sub["Begin Time (s)"] - query_breakpoint + support_breakpoint
            query_st_sub["End Time (s)"] = query_st_sub["End Time (s)"] - query_breakpoint + support_breakpoint
            
            new_st = pd.concat([support_st_sub, query_st_sub])
            
            new_audio_fp = os.path.join(new_audio_dir, "crossfile_" + os.path.basename(row["audio_fp"]))
            sf.write(new_audio_fp, new_audio, sr)
            
            new_st_fp = os.path.join(new_st_dir, "crossfile_" + os.path.basename(row['selection_table_fp']))
            new_st.to_csv(new_st_fp, sep = '\t', index=False)
            
            new_manifest['audio_fp'].append(new_audio_fp.split("/formatted/")[1])
            new_manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
    new_manifest = pd.DataFrame(new_manifest)
    new_manifest.to_csv(os.path.join(new_dataset_dir, "manifest.csv"), index=False)
        
    