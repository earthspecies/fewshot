from glob import glob
import os
import shutil
import pandas as pd

dataset_names = ["marmoset", "carrion_crow", "katydid", "Spiders", "rana_sierrae", "Powdermill", "Anuraset", "Hawaii", "right_whale", "gibbons", "gunshots", "humpback", "ruffed_grouse", "PW", "PB", "PB24", "ME", "HB", "RD"]

source_dir = '/home/jupyter/data/fewshot_data/evaluation/bboxes'
target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
old_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted_nobbox'

for dataset in dataset_names:
    fps = sorted(glob(os.path.join(source_dir, dataset, "selection_tables", "*_AC.txt")))
    dd = os.path.join(target_dir, dataset, "selection_tables")
    if not os.path.exists(dd):
        os.makedirs(dd)
    for fp in fps:
        st = pd.read_csv(fp, sep='\t')
        
        if ("Channel" in st.columns) and (len(st["Channel"].unique()) > 1):
            st = st[st['Channel'] == 1]
        if "Annotation" not in st.columns:
            st["Annotation"] = "POS"
        
        target_fn = os.path.basename(fp).replace("_AC.txt", ".txt")
        target_fp = os.path.join(target_dir, dataset, "selection_tables", target_fn)
        
        st.to_csv(target_fp, index=False, sep='\t')
        # shutil.copy(fp, target_fp)
    print(dataset, len(fps), len(sorted(glob(os.path.join(old_dir, dataset, "selection_tables", "*.txt")))))