import os
import pandas as pd
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

birdnet_embeddings = sorted(glob('/home/jupyter/data/fewshot_data/data_large/birdnet_embeddings_chunk_*.pkl'))

splits = ["Train", "Val", "Test"]
labels = ["Include", "Exclude"]
qf_embs = {}
for split in splits:
    qf_embs[split] = {}
    for label in labels:
        qf_embs[split][label] = {}

checksum0 = 0
        
for birdnet_embedding in tqdm(birdnet_embeddings):
    with open(birdnet_embedding, 'rb') as f:
        emb = pickle.load(f)
    
    for split in splits:
        for label in labels:
            print(split, label)
            qf_files = sorted(set(glob(f'/home/jupyter/data/fewshot_data/quality_filter/{split}_{label}/*.wav')))
            
            for qf_file in tqdm(qf_files):
                qf_fn = os.path.basename(qf_file)
                qf_key = os.path.join('/home/davidrobinson/fewshot_data/data_large/animalspeak_pseudovox', qf_fn)
                if qf_key in emb:
                    qf_embs[split][label][qf_key] = emb[qf_key]
                    checksum0 += 1
                    
checksum = 0
for split in splits:
    for label in labels:
        checksum += len(qf_embs[split][label].keys())
        
if checksum != checksum0:
    import pdb; pdb.set_trace()
            
out_fp = '/home/jupyter/data/fewshot_data/quality_filter/birdnet_embeddings_qf.pkl'
with open(out_fp, 'wb') as f:
    pickle.dump(qf_embs, f)