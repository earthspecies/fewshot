import os
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
from glob import glob
from tqdm import tqdm

embeddings_fp = '/home/jupyter/data/fewshot_data/quality_filter/birdnet_embeddings_qf.pkl'

with open(embeddings_fp, 'rb') as f:
    emb = pickle.load(f)
    
train_X = []
train_y = []
train_names = []

test_X = []
test_y = []
test_names = []

val_X = []
val_y = []
val_names = []

for label in ["Include", "Exclude"]:
    for k in emb['Train'][label]:
        x = emb['Train'][label][k]
        x = np.array([t['embeddings'] for t in x])
        x = np.mean(x, axis = 0)
        train_X.append(x)
        train_names.append(k)
        
        train_y.append(True if label=="Exclude" else False)
        
train_X = np.stack(train_X)

for label in ["Include", "Exclude"]:
    for k in emb['Val'][label]:
        x = emb['Val'][label][k]
        x = np.array([t['embeddings'] for t in x])
        x = np.mean(x, axis = 0)
        val_X.append(x)
        val_names.append(k)
        
        val_y.append(True if label=="Exclude" else False)
        
val_X = np.stack(val_X)
        
for label in ["Include", "Exclude"]:
    for k in emb['Test'][label]:
        x = emb['Test'][label][k]
        x = np.array([t['embeddings'] for t in x])
        x = np.mean(x, axis = 0)
        test_X.append(x)
        test_names.append(k)
        
        test_y.append(True if label=="Exclude" else False)
        
test_X = np.stack(test_X)

LR = LogisticRegression(max_iter=10000)
LR.fit(train_X, train_y)

preds = LR.predict(test_X)

acc = (preds == test_y).sum() / len(test_y)
print("Accuracy:")
print(acc)

fp = (preds & ~np.array(test_y)).sum()
fn = (~preds & np.array(test_y)).sum()

print("False Positive Rate (chances incorrectly excluded)")
print(fp/len(test_y))

print("False Negative Rate (chances incorrectly included)")
print(fn/len(test_y))

out_fp = '/home/jupyter/data/fewshot_data/quality_filter/qf_model.pkl'
with open(out_fp, 'wb') as f:
    pickle.dump(LR, f)
    
print(f"Saved model to {out_fp}")

'''
Accuracy:
0.85
False Positive Rate (chances incorrectly excluded)
0.11666666666666667
False Negative Rate (chances incorrectly included)
0.03333333333333333
Saved model to /home/jupyter/data/fewshot_data/quality_filter/qf_model.pkl
'''