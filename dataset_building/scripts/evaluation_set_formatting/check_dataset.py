import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np

dataset_names = ["audioset_strong", "DESED", "marmoset", "carrion_crow", "katydid", "Spiders", "rana_sierrae", "Powdermill", "Anuraset", "Hawaii", "right_whale", "gibbons", "gunshots", "humpback", "ruffed_grouse"]

formatted_dataset_parent_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted"
for dataset in dataset_names:
    
    print(f"Dataset: {dataset}" + "="*75)
    
    data_dir = os.path.join(formatted_dataset_parent_dir, dataset)
    manifest = pd.read_csv(os.path.join(data_dir, "manifest.csv"))
    
    for i, row in manifest.iterrows():
        audio_fp = os.path.join(formatted_dataset_parent_dir, row["audio_fp"])
        st_fp = os.path.join(formatted_dataset_parent_dir, row["selection_table_fp"])
        
        assert os.path.exists(audio_fp)
        assert os.path.exists(st_fp)
        
        st = pd.read_csv(st_fp, sep='\t')
        
        print(os.path.basename(st_fp))
        
        print("Annotations present:")
        
        print(sorted(st["Annotation"].unique()))
        
        n_annots = len(st[st["Annotation"] != "Unknown"])
        assert n_annots > 5
        
        print(f"There are {n_annots} annotations")
        
        st_sub = st[st["Annotation"] != "Unknown"]
        st_sub["Duration"] = st_sub["End Time (s)"] - st_sub["Begin Time (s)"]
        
        assert np.all(st_sub["Duration"] > 0.01)
        
        # check no overlaps of single class
        for anno in sorted(st["Annotation"].unique()):
            if anno == "Unknown":
                continue
            for i, row in st_sub.iterrows():
                overlap1 = st_sub[(st_sub["End Time (s)"] > row["Begin Time (s)"]) & (st_sub["Begin Time (s)"] < row["Begin Time (s)"])]
                overlap2 = st_sub[(row["End Time (s)"] > st_sub["Begin Time (s)"]) & (row["End Time (s)"] < st_sub["End Time (s)"])]
                overlap3 = st_sub[(st_sub["Begin Time (s)"] > row["Begin Time (s)"]) & (st_sub["End Time (s)"] < row["End Time (s)"])]
                assert len(overlap1) + len(overlap2) + len(overlap3) == 0
        
        
        print("Event Lengths:")
        
        print(f' Mean: {st_sub["Duration"].mean()}, STD: {st_sub["Duration"].std()}, min: {st_sub["Duration"].min()}, max: {st_sub["Duration"].max()}')
        print("-"*50)
        
      