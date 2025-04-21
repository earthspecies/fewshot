import os
import pandas as pd
from glob import glob

new_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
old_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted_nobbox'

dataset_names = ["marmoset", "carrion_crow", "katydid", "Spiders", "rana_sierrae", "Powdermill", "Anuraset", "Hawaii", "right_whale", "gibbons", "gunshots", "humpback", "ruffed_grouse"]
dataset_names = [x+"_crossfile" for x in dataset_names]

for dataset in dataset_names:
    print(dataset)
    fps = sorted(glob(os.path.join(new_dir, dataset, "selection_tables", "*.txt")))
    for fp in fps:
        new_st = pd.read_csv(fp, sep='\t')
        old_fp = os.path.join(old_dir, dataset, "selection_tables", os.path.basename(fp))
        old_st = pd.read_csv(old_fp, sep='\t')
        
        if len(new_st) != len(old_st):
            print(new_st, len(new_st), len(old_st))
    
