import os
import pandas as pd
import librosa
import soundfile as sf
import shutil

dataset_names = ["PB24", "PB", "marmoset", "Spiders"] + [x+"_crossfile" for x in ["marmoset", "Spiders"]]
parent_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted"

for dataset in dataset_names:
    print(f"Processing {dataset}")
    dataset_dir = os.path.join(parent_dir, dataset)
    new_dataset_dir = os.path.join(parent_dir, dataset+"_halftime")
    
    if os.path.exists(new_dataset_dir):
        shutil.rmtree(new_dataset_dir)
    
    new_audio_dir = os.path.join(new_dataset_dir, "audio")
    new_st_dir = os.path.join(new_dataset_dir, "selection_tables")
    
    for d in [new_audio_dir, new_st_dir]:
        os.makedirs(d)
    
    manifest_fp = os.path.join(dataset_dir, "manifest.csv")
    manifest=pd.read_csv(manifest_fp)
    new_manifest = {"audio_fp" : [], "selection_table_fp" : []}
    
    for i, row in manifest.iterrows():
        audio, sr = sf.read(os.path.join(parent_dir, row["audio_fp"]))
        new_sr = sr // 2
        
        new_audio_fp = os.path.join(new_audio_dir, os.path.basename(row["audio_fp"]))
        sf.write(new_audio_fp, audio, new_sr)
        
        st = pd.read_csv(os.path.join(parent_dir, row["selection_table_fp"]), sep='\t')
        st["Begin Time (s)"] = st["Begin Time (s)"]*2
        st["End Time (s)"] = st["End Time (s)"]*2
        
        st["Low Freq (Hz)"] = st["Low Freq (Hz)"]/2
        st["High Freq (Hz)"] = st["High Freq (Hz)"]/2
        
        new_st_fp = os.path.join(new_st_dir, os.path.basename(row['selection_table_fp']))
        st.to_csv(new_st_fp, sep='\t', index=False)
        
        new_manifest['audio_fp'].append(new_audio_fp.split("/formatted/")[1])
        new_manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    new_manifest = pd.DataFrame(new_manifest)
    new_manifest.to_csv(os.path.join(new_dataset_dir, "manifest.csv"), index=False)

dataset_names = ["katydid", "katydid_crossfile"]
parent_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted"

for dataset in dataset_names:
    print(f"Processing {dataset}")
    dataset_dir = os.path.join(parent_dir, dataset)
    new_dataset_dir = os.path.join(parent_dir, dataset+"_sixthtime")
    
    if os.path.exists(new_dataset_dir):
        shutil.rmtree(new_dataset_dir)
    
    new_audio_dir = os.path.join(new_dataset_dir, "audio")
    new_st_dir = os.path.join(new_dataset_dir, "selection_tables")
    
    for d in [new_audio_dir, new_st_dir]:
        os.makedirs(d)
    
    manifest_fp = os.path.join(dataset_dir, "manifest.csv")
    manifest=pd.read_csv(manifest_fp)
    new_manifest = {"audio_fp" : [], "selection_table_fp" : []}
    
    for i, row in manifest.iterrows():
        audio, sr = sf.read(os.path.join(parent_dir, row["audio_fp"]))
        new_sr = sr // 6
        
        new_audio_fp = os.path.join(new_audio_dir, os.path.basename(row["audio_fp"]))
        sf.write(new_audio_fp, audio, new_sr)
        
        st = pd.read_csv(os.path.join(parent_dir, row["selection_table_fp"]), sep='\t')
        st["Begin Time (s)"] = st["Begin Time (s)"]*6
        st["End Time (s)"] = st["End Time (s)"]*6
        
        st["Low Freq (Hz)"] = st["Low Freq (Hz)"]/6
        st["High Freq (Hz)"] = st["High Freq (Hz)"]/6
        
        new_st_fp = os.path.join(new_st_dir, os.path.basename(row['selection_table_fp']))
        st.to_csv(new_st_fp, sep='\t', index=False)
        
        new_manifest['audio_fp'].append(new_audio_fp.split("/formatted/")[1])
        new_manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    new_manifest = pd.DataFrame(new_manifest)
    new_manifest.to_csv(os.path.join(new_dataset_dir, "manifest.csv"), index=False)