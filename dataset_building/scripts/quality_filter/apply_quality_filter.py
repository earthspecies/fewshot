import os
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
from glob import glob
from tqdm import tqdm

birdnet_embeddings = sorted(glob('/home/jupyter/data/fewshot_data/data_large/birdnet_embeddings_chunk_*.pkl'))

model_fp = '/home/jupyter/data/fewshot_data/quality_filter/qf_model.pkl'
with open(model_fp, 'rb') as f:
    model = pickle.load(f)
    
df = {"filepath" : [], "exclude" : []}
        
for birdnet_embedding in tqdm(birdnet_embeddings):
    with open(birdnet_embedding, 'rb') as f:
        emb = pickle.load(f)
    
    for k in tqdm(sorted(emb.keys())):
        x = emb[k]
        x = np.array([t['embeddings'] for t in x])
        x = np.mean(x, axis = 0)
        p = model.predict(x.reshape(1, -1))[0]
        df['filepath'].append(k)
        df['exclude'].append(p)
        
df = pd.DataFrame(df)
df.to_csv('/home/jupyter/data/fewshot_data/data_large/quality_filtered.csv', index=False)
