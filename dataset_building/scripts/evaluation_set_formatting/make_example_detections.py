import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import patches
import seaborn as sns


datasets = ["marmoset_halftime", "gunshots", "Anuraset", "carrion_crow", "katydid_sixthtime", "Spiders_halftime", "rana_sierrae", "Powdermill",  "Hawaii", "right_whale", "gibbons", "humpback", "ruffed_grouse"]

dataset_to_dataset_name = {"marmoset_halftime" : "MS",
                           "Anuraset" : "AS",
                           "carrion_crow" : "CC",
                           "katydid_sixthtime" : "KD",
                           "Spiders_halftime" : "JS", 
                           "rana_sierrae" : "RS", 
                           "Powdermill" : "PM", 
                           "Hawaii" : "HA",
                           "right_whale" : "RW", 
                           "gibbons" : "HG",
                           "gunshots" : "GS", 
                           "humpback" : "HW",
                           "ruffed_grouse" : "RG"
                          }

# dataset_names = [dataset_to_dataset_name[x] for x in datasets]
# datasets = [x for _,x in sorted(zip(dataset_names, datasets))]

formatted_dataset_parent_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted_for_dcase"
preds_dir = '/home/jupyter/drasdic/results/drasdic/test_outputs_X'

outs_dir = os.path.join("/home/jupyter/drasdic/results/drasdic/example_preds")
if not os.path.exists(outs_dir):
    os.makedirs(outs_dir)
    
rng = np.random.default_rng(0)

N = 100

for fign in range(N):

    fig, ax = plt.subplots(2,1, figsize=(12,5))
    
    dataset = rng.choice(datasets)
    
    pred_dir = os.path.join(preds_dir, dataset)
    anno_dir = os.path.join(formatted_dataset_parent_dir, dataset)
    
    chosen_anno_fp = rng.choice(sorted(glob(os.path.join(anno_dir, "*.csv"))))
    chosen_anno_fn = os.path.basename(chosen_anno_fp)
    chosen_pred_fn = chosen_anno_fn
    chosen_pred_fp = os.path.join(pred_dir, chosen_pred_fn).replace(".csv", ".txt")
    
    audio_fp = chosen_anno_fp.replace(".csv", ".wav")
    
    pred_st = pd.read_csv(chosen_pred_fp, sep='\t')
    anno_st = pd.read_csv(chosen_anno_fp)
    anno_st["Begin Time (s)"] = anno_st["Starttime"]
    anno_st["End Time (s)"] = anno_st["Endtime"]
    anno_st["Annotation"] = anno_st["Q"]
    
    anno_st = anno_st[anno_st["Annotation"] == "POS"].copy()
    # anno_st["Annotation"] == "POS"
    
    breakpoint = sorted(anno_st["End Time (s)"])[4]
    
    query_anno_st = anno_st[anno_st["Begin Time (s)"] > breakpoint]
    query_pred_st = anno_st[anno_st["Begin Time (s)"] > breakpoint]
    
    bothsts = pd.concat([query_anno_st, query_pred_st])
    example_start = max(0, rng.choice(list(bothsts["Begin Time (s)"])) - 1)
    
    y, sr = librosa.load(audio_fp, offset=example_start, duration = 10, sr = None)
    
    fmax = int(sr*3/8)#sr//2 #min(10000, sr//2)
    hop_length = 1024
    D = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=fmax, fmin=10), ref=np.max)
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    imgref = librosa.display.specshow(D, y_axis=None, x_axis='time', sr=sr, ax=ax[0], fmax=fmax, fmin=100)
    imgpred = librosa.display.specshow(D, y_axis=None, x_axis='time', sr=sr, ax=ax[1], fmax=fmax, fmin=100)
    
#     st = anno_st
#     st_sub = st[((st["Begin Time (s)"] >= example_start) & (st["Begin Time (s)"] <= example_start+10)) | ((st["End Time (s)"] >= example_start) & (st["End Time (s)"] <= example_start+10))]
#     for _, row in st_sub.iterrows():
#         boxstart = max(0, row["Begin Time (s)"] - example_start)
#         boxend = min(row["End Time (s)"] - example_start, 10)

#         ax[0].axvspan(boxstart, boxend, color='white', alpha=0.2, label='Event of Interest', linewidth=2)
        
#     st = pred_st
#     st_sub = st[((st["Begin Time (s)"] >= example_start) & (st["Begin Time (s)"] <= example_start+10)) | ((st["End Time (s)"] >= example_start) & (st["End Time (s)"] <= example_start+10))]
#     for _, row in st_sub.iterrows():
#         boxstart = max(0, row["Begin Time (s)"] - example_start)
#         boxend = min(row["End Time (s)"] - example_start, 10)

#         ax[1].axvspan(boxstart, boxend, color='yellow', alpha=0.2, label='Event of Interest', linewidth=2)

#     ax[0].set_ylabel(f"Labels ({dataset_to_dataset_name[dataset]})", fontsize=20)
#     ax[0].set_xticks([])
#     ax[0].set_xlabel("")
#     ax[1].set_ylabel(f"Predictions", fontsize=20)
#     ax[1].set_xticks([])
#     ax[1].set_xlabel("")

    
    ax[0].set_ylabel(f"{dataset_to_dataset_name[dataset]}", fontsize=20)
    ax[0].set_xticks([])
    ax[0].set_xlabel("")
    ax[1].set_ylabel("", fontsize=20)
    ax[1].set_xticks([])
    ax[1].set_xlabel("")
                

    plt.tight_layout()
    plt.savefig(os.path.join(outs_dir, f"{fign}_{dataset}.png"))
    plt.close()

