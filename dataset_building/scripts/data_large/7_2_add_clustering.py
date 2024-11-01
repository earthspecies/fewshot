import os
import pandas as pd
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans

embeddings_saved = '/home/jupyter/data/fewshot_data/data_large/embeddings_saved.npy'
all_embeddings_saved_inorder = '/home/jupyter/data/fewshot_data/data_large/all_embeddings_saved_inorder.npy'
clustering_model_sizes = [256,128,64,32,16,8]
clustering_model_fps = {i : f'/home/jupyter/data/fewshot_data/data_large/clustering_model_{i}.pkl' for i in clustering_model_sizes}
birdnet_embeddings = sorted(glob('/home/jupyter/data/fewshot_data/data_large/birdnet_embeddings_chunk_*.pkl'))
info_fp = '/home/jupyter/data/fewshot_data/data_large/animalspeak_pseudovox_with_birdnet_with_qf.csv'

info_df = pd.read_csv(info_fp).reset_index(drop=True)  

# if embeddings_saved is None:
#     embeddings = []
#     for birdnet_embedding in tqdm(birdnet_embeddings):
#         with open(birdnet_embedding, 'rb') as f:
#             emb = pickle.load(f)
        
#         for i, row in tqdm(info_df.iterrows()):
#             if row['qf_exclude']:
#                 continue
#             k = row['pseudovox_audio_fp']
#             if k in emb:
#                 x = emb[k]
#                 x = np.array([t['embeddings'] for t in x])
#                 x = np.mean(x, axis = 0, dtype=np.float32)
#                 embeddings.append(x)
                
#     emb = None
                
#     embeddings = np.stack(embeddings)
#     np.save('/home/jupyter/data/fewshot_data/data_large/embeddings_saved.npy', embeddings)
        
# else:
#     embeddings = np.load(embeddings_saved)
    
clustering_models = {}
    
for s in clustering_model_sizes:
    if clustering_model_fps[s] is None:
        n_clusters = np.shape(embeddings)[0] // s
        print(f"Fitting clustering model with mean cluster size {s}, total clusters {n_clusters}")
        clustering_model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4*4096, verbose=True, init= 'random',n_init=10) #embeddings[:n_clusters,:]
        
        clustering_model.fit(embeddings)
        with open(f'/home/jupyter/data/fewshot_data/data_large/clustering_model_{s}.pkl', 'wb') as f:
            pickle.dump(clustering_model, f)
            
        clustering_models[s] = clustering_model
    else:
        with open(clustering_model_fps[s], 'rb') as f:
            clustering_models[s] = pickle.load(f)
            
embeddings = None

if all_embeddings_saved_inorder is None:
    all_embeddings = np.zeros((len(info_df), 1024))
    for birdnet_embedding in tqdm(birdnet_embeddings):
        with open(birdnet_embedding, 'rb') as f:
            emb = pickle.load(f)
        
        for i, row in tqdm(info_df.iterrows()):
            k = row['pseudovox_audio_fp']
            if k in emb:
                x = emb[k]
                x = np.array([t['embeddings'] for t in x])
                x = np.mean(x, axis = 0, dtype=np.float32)
                all_embeddings[i,:]=x
                
    emb = None
    
    np.save('/home/jupyter/data/fewshot_data/data_large/all_embeddings_saved_inorder.npy', all_embeddings)
else:
    all_embeddings = np.load(all_embeddings_saved_inorder) 
    
for s in clustering_model_sizes:
    clustering_models[s].cluster_centers_ = clustering_models[s].cluster_centers_.astype(float) #workaround https://github.com/pycaret/pycaret/issues/3774
    
    c = []
    batchsize = 1024*4

    for chunk in tqdm(np.arange(0, len(all_embeddings), batchsize)):
        x = all_embeddings[int(chunk):int(chunk)+batchsize,:]
        c.extend(list(clustering_models[s].predict(x)))
            
    info_df[f'c_{s}'] = pd.Series(c)
    info_df.to_csv('/home/jupyter/data/fewshot_data/data_large/animalspeak_pseudovox_with_birdnet_with_qf_with_c.csv', index=False)
