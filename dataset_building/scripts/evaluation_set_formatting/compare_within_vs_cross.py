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

def get_breakpoint(st, break_after = 5):
    st_sub = st[st["Annotation"] != "Unknown"]
    endtime = sorted(st_sub["End Time (s)"])[break_after-1]
    return endtime

def get_mean_spectrum(audio_fp, st):
    audios = []
    for k, event in st.iterrows():
        audio, sr = librosa.load(audio_fp, offset = event["Begin Time (s)"], duration = event["End Time (s)"] - event["Begin Time (s)"], sr = None, mono=True)
        audios.append(audio)
    audios = np.concatenate(audios)
    spectrogram = librosa.feature.melspectrogram(y=audios, sr=sr, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', power=2.0, n_mels=256)
    spectrum = np.mean(spectrogram, axis = -1)
    return spectrum

datasets = ["marmoset", "Anuraset", "carrion_crow", "katydid", "Spiders", "rana_sierrae", "Powdermill",  "Hawaii", "right_whale", "gibbons", "gunshots", "humpback", "ruffed_grouse"]

dataset_to_dataset_name = {"marmoset" : "MS",
                           "Anuraset" : "AS",
                           "carrion_crow" : "CC",
                           "katydid" : "KD",
                           "Spiders" : "JS", 
                           "rana_sierrae" : "RS", 
                           "Powdermill" : "PM", 
                           "Hawaii" : "HA",
                           "right_whale" : "RW", 
                           "gibbons" : "HG",
                           "gunshots" : "GS", 
                           "humpback" : "HW",
                           "ruffed_grouse" : "RG"
                          }

dataset_names = [dataset_to_dataset_name[x] for x in datasets]
datasets = [x for _,x in sorted(zip(dataset_names, datasets))]

formatted_dataset_parent_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted"

### EVENT SIMILARITY

event_similarity = {"within_file" : [], "cross_file" : [], "Dataset" : []}

for i, dataset in enumerate(datasets):
    print(f"Processing {dataset}")
    
    data_dir = os.path.join(formatted_dataset_parent_dir, dataset)
    manifest = pd.read_csv(os.path.join(data_dir, "manifest.csv"))
    
    for j, row in manifest.iterrows():
        # within file version
        audio_fp = os.path.join(formatted_dataset_parent_dir, row["audio_fp"])
        st_fp = os.path.join(formatted_dataset_parent_dir, row["selection_table_fp"])
        
        st = pd.read_csv(st_fp, sep = "\t")
        st = st[st["Annotation"] != "Unknown"].sort_values("End Time (s)")
        
        audio_duration = librosa.get_duration(path = audio_fp)
        
        breakpoint = get_breakpoint(st)
        st_support = st[st["End Time (s)"] <= breakpoint]
        st_query = st[st["End Time (s)"] > breakpoint]
        
        support_event_spectrum = get_mean_spectrum(audio_fp, st_support)
        query_event_spectrum = get_mean_spectrum(audio_fp, st_query)
        within_file_correlation = np.dot(support_event_spectrum, query_event_spectrum) / (np.linalg.norm(support_event_spectrum)*np.linalg.norm(query_event_spectrum))
        
        
        # cross file version
        audio_fp = row["audio_fp"].replace(f"{dataset}/audio/", f"{dataset}_crossfile/audio/crossfile_")
        audio_fp = os.path.join(formatted_dataset_parent_dir, audio_fp)
        st_fp = row["selection_table_fp"].replace(f"{dataset}/selection_tables/", f"{dataset}_crossfile/selection_tables/crossfile_")
        st_fp = os.path.join(formatted_dataset_parent_dir, st_fp)
        
        st = pd.read_csv(st_fp, sep = "\t")
        st = st[st["Annotation"] != "Unknown"].sort_values("End Time (s)")
        
        audio_duration = librosa.get_duration(path = audio_fp)
        
        breakpoint = get_breakpoint(st)
        st_support = st[st["End Time (s)"] <= breakpoint]
        st_query = st[st["End Time (s)"] > breakpoint]
        
        support_event_spectrum = get_mean_spectrum(audio_fp, st_support)
        query_event_spectrum = get_mean_spectrum(audio_fp, st_query)
        cross_file_correlation = np.dot(support_event_spectrum, query_event_spectrum) / (np.linalg.norm(support_event_spectrum)*np.linalg.norm(query_event_spectrum))
        
        event_similarity["within_file"].append(within_file_correlation)
        event_similarity["cross_file"].append(cross_file_correlation)
        event_similarity["Dataset"].append(dataset_to_dataset_name[dataset])
        
event_similarity = pd.DataFrame(event_similarity)
with sns.color_palette("colorblind"):
    ax = sns.scatterplot(data=event_similarity, x = "within_file", y = "cross_file", hue = "Dataset", alpha = 0.5, s=20)
print(sum(event_similarity["within_file"] > event_similarity["cross_file"]) / len(event_similarity))
ax.set_xlabel("Within-recording support-query similarity")
ax.set_ylabel("Cross-recording support-query similarity")
ax.set_title("Event similarity, for within- and cross-recording versions")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.plot([0,1], [0,1], alpha = 0.2)
plt.savefig(os.path.join(formatted_dataset_parent_dir, "event_similarity_comparison.png"))
plt.close()



#### BACKGROUND SIMILARITY

background_similarity = {"within_file" : [], "cross_file" : [], "Dataset" : []}

for i, dataset in enumerate(datasets):
    print(f"Processing {dataset}")
    
    data_dir = os.path.join(formatted_dataset_parent_dir, dataset)
    manifest = pd.read_csv(os.path.join(data_dir, "manifest.csv"))
    
    for j, row in manifest.iterrows():
        # within file version
        audio_fp = os.path.join(formatted_dataset_parent_dir, row["audio_fp"])
        st_fp = os.path.join(formatted_dataset_parent_dir, row["selection_table_fp"])
        
        st = pd.read_csv(st_fp, sep = "\t")
        st = st[st["Annotation"] != "Unknown"].sort_values("End Time (s)")
        
        audio_duration = librosa.get_duration(path = audio_fp)
        
        breakpoint = get_breakpoint(st)
        st_support = st[st["End Time (s)"] <= breakpoint]
        st_query = st[st["End Time (s)"] > breakpoint]
        
        st_support_background = st_support.copy().reset_index()
        st_support_background["Begin Time (s)"] = pd.Series([0] + list(st_support["End Time (s)"]))
        st_support_background["End Time (s)"] = pd.Series(list(st_support["Begin Time (s)"]) + [breakpoint])
        st_support_background["Duration"] = st_support_background["End Time (s)"] - st_support_background["Begin Time (s)"]
        
        # print(f"Omitting {sum(st_support_background['Duration'] <= 0)} chunks")
        st_support_background = st_support_background[st_support_background["Duration"] > 0]
        st_support_background["Begin Time (s)"] = np.maximum(st_support_background["Begin Time (s)"], st_support_background["Begin Time (s)"] + st_support_background["Duration"]/2 - 0.5)
        st_support_background["End Time (s)"] = np.minimum(st_support_background["End Time (s)"], st_support_background["End Time (s)"] - st_support_background["Duration"]/2 + 0.5)
        st_support_background["Duration"] = st_support_background["End Time (s)"] - st_support_background["Begin Time (s)"]
        
        st_query_background = st_query.copy().reset_index()
        st_query_background["Begin Time (s)"] = pd.Series([breakpoint] + list(st_query["End Time (s)"]))
        st_query_background["End Time (s)"] = pd.Series(list(st_query["Begin Time (s)"]) + [audio_duration])
        st_query_background["Duration"] = st_query_background["End Time (s)"] - st_query_background["Begin Time (s)"]
        
        # print(f"Omitting {sum(st_query_background['Duration'] <= 0)} chunks")
        st_query_background = st_query_background[st_query_background["Duration"] > 0]
        st_query_background["Begin Time (s)"] = np.maximum(st_query_background["Begin Time (s)"], st_query_background["Begin Time (s)"] + st_query_background["Duration"]/2 - 0.5)
        st_query_background["End Time (s)"] = np.minimum(st_query_background["End Time (s)"], st_query_background["End Time (s)"] - st_query_background["Duration"]/2 + 0.5)
        st_query_background["Duration"] = st_query_background["End Time (s)"] - st_query_background["Begin Time (s)"]
        
        support_background_spectrum = get_mean_spectrum(audio_fp, st_support_background)
        query_background_spectrum = get_mean_spectrum(audio_fp, st_query_background)
        within_file_correlation = np.dot(support_background_spectrum, query_background_spectrum) / (np.linalg.norm(support_background_spectrum)*np.linalg.norm(query_background_spectrum))
        
        # cross file version
        audio_fp = row["audio_fp"].replace(f"{dataset}/audio/", f"{dataset}_crossfile/audio/crossfile_")
        audio_fp = os.path.join(formatted_dataset_parent_dir, audio_fp)
        st_fp = row["selection_table_fp"].replace(f"{dataset}/selection_tables/", f"{dataset}_crossfile/selection_tables/crossfile_")
        st_fp = os.path.join(formatted_dataset_parent_dir, st_fp)
        st = pd.read_csv(st_fp, sep = "\t")
        st = st[st["Annotation"] != "Unknown"].sort_values("End Time (s)")
        
        audio_duration = librosa.get_duration(path = audio_fp)
        
        breakpoint = get_breakpoint(st)
        st_support = st[st["End Time (s)"] <= breakpoint]
        st_query = st[st["End Time (s)"] > breakpoint]
        
        st_support_background = st_support.copy().reset_index()
        st_support_background["Begin Time (s)"] = pd.Series([0] + list(st_support["End Time (s)"]))
        st_support_background["End Time (s)"] = pd.Series(list(st_support["Begin Time (s)"]) + [breakpoint])
        st_support_background["Duration"] = st_support_background["End Time (s)"] - st_support_background["Begin Time (s)"]
        
        # print(f"Omitting {sum(st_support_background['Duration'] <= 0)} chunks")
        st_support_background = st_support_background[st_support_background["Duration"] > 0]
        st_support_background["Begin Time (s)"] = np.maximum(st_support_background["Begin Time (s)"], st_support_background["Begin Time (s)"] + st_support_background["Duration"]/2 - 0.5)
        st_support_background["End Time (s)"] = np.minimum(st_support_background["End Time (s)"], st_support_background["End Time (s)"] - st_support_background["Duration"]/2 + 0.5)
        st_support_background["Duration"] = st_support_background["End Time (s)"] - st_support_background["Begin Time (s)"]
        
        st_query_background = st_query.copy().reset_index()
        st_query_background["Begin Time (s)"] = pd.Series([breakpoint] + list(st_query["End Time (s)"]))
        st_query_background["End Time (s)"] = pd.Series(list(st_query["Begin Time (s)"]) + [audio_duration])
        st_query_background["Duration"] = st_query_background["End Time (s)"] - st_query_background["Begin Time (s)"]
        
        # print(f"Omitting {sum(st_query_background['Duration'] <= 0)} chunks")
        st_query_background = st_query_background[st_query_background["Duration"] > 0]
        st_query_background["Begin Time (s)"] = np.maximum(st_query_background["Begin Time (s)"], st_query_background["Begin Time (s)"] + st_query_background["Duration"]/2 - 0.5)
        st_query_background["End Time (s)"] = np.minimum(st_query_background["End Time (s)"], st_query_background["End Time (s)"] - st_query_background["Duration"]/2 + 0.5)
        st_query_background["Duration"] = st_query_background["End Time (s)"] - st_query_background["Begin Time (s)"]
        
        support_background_spectrum = get_mean_spectrum(audio_fp, st_support_background)
        query_background_spectrum = get_mean_spectrum(audio_fp, st_query_background)
        
        cross_file_correlation = np.dot(support_background_spectrum, query_background_spectrum) / (np.linalg.norm(support_background_spectrum)*np.linalg.norm(query_background_spectrum))
        
        background_similarity["within_file"].append(within_file_correlation)
        background_similarity["cross_file"].append(cross_file_correlation)
        background_similarity["Dataset"].append(dataset_to_dataset_name[dataset])
        
background_similarity = pd.DataFrame(background_similarity)
with sns.color_palette("colorblind"):
    ax = sns.scatterplot(data=background_similarity, x = "within_file", y = "cross_file", hue = "Dataset", alpha = 0.5, s=20)
print(sum(event_similarity["within_file"] > event_similarity["cross_file"]) / len(event_similarity))
print(sum(background_similarity["within_file"] > background_similarity["cross_file"]) / len(background_similarity))
ax.set_xlabel("Within-recording support-query similarity")
ax.set_ylabel("Cross-recording support-query similarity")
ax.set_title("Background similarity, for within- and cross-recording versions")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.plot([0,1], [0,1], alpha = 0.2)
plt.savefig(os.path.join(formatted_dataset_parent_dir, "background_similarity_comparison.png"))
plt.close()