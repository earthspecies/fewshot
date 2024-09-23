'''
Script to process original DESED data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/DESED/dataset/'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    rng = np.random.default_rng(0)
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "DESED"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    if os.path.exists(dtgt_dir):
        shutil.rmtree(dtgt_dir)
    
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    
    annotations = pd.read_csv(os.path.join(DATA_DIR, 'metadata/eval/public.tsv'), sep='\t')
    annotations["Duration"] = annotations["offset"] - annotations["onset"]
    annotations = annotations[annotations["Duration"] <9.8]
    
    
    annotation_count = {"anno" : [], "count" : []}
    for annotation in annotations["event_label"].unique():
        annot_sub = annotations[annotations["event_label"] == annotation]
        annotation_count["anno"].append(annotation)
        annotation_count["count"].append(len(annot_sub))
        
    annotation_count = pd.DataFrame(annotation_count)
    annotation_count = annotation_count[annotation_count["count"] > 15] # require at least 15 files per anno type
    
    valid_annos = sorted(rng.permutation(sorted(annotation_count["anno"]))[:100]) # use up to 100 labels, chosen randomly
    
    for anno in valid_annos:
        print(f"Processing {anno} files")
        annot_sub = annotations[annotations["event_label"] == anno]
        valid_files = annot_sub["filename"].unique()
        
        for i in range(5): # create 5 examples per annotation type
            files_to_use = rng.permutation(valid_files)[:6] # grab 6 files per example (total of 1 minute per example)
            
            audio = []
            st = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : [], "Original Filename" : []}
            shift = 0
            
            for file in files_to_use:
                audio_fp = os.path.join(DATA_DIR, "audio/eval/public/", file)
                a, sr = librosa.load(audio_fp, sr=16000, mono=True)
                
                audio.append(a)
                
                selections = annot_sub[annot_sub['filename'] == file]
                st["Begin Time (s)"].extend(selections["onset"] + shift)
                st["End Time (s)"].extend(selections["offset"] + shift)
                st["Annotation"].extend(selections["event_label"])
                st["Original Filename"].extend([file for _ in range(len(selections))])
                shift += len(a) / sr
                
            audio = np.concatenate(audio)
            st = pd.DataFrame(st)
            
            new_audio_fn = f"{anno}_{i}.wav"
            new_anno_fn = f"{anno}_{i}.txt"
            
            new_audio_fp = os.path.join(audio_tgt_dir, new_audio_fn)
            new_st_fp = os.path.join(annot_tgt_dir, new_anno_fn)
            
            sf.write(new_audio_fp, audio, sr)
            st.to_csv(new_st_fp, sep='\t', index=False)
            
            manifest["audio_fp"].append(new_audio_fp.split("/formatted/")[1])
            manifest["selection_table_fp"].append(new_st_fp.split("/formatted/")[1])  
            
    
#     for i, audio_fp in enumerate(sorted(glob(os.path.join(DATA_DIR, "audio/eval/public/*.wav")))):
#         print(f"Processing {audio_fp}")
        
#         shutil.copy(audio_fp, audio_tgt_dir)
#         manifest["audio_fp"].append(os.path.join(audio_tgt_dir, os.path.basename(audio_fp)).split("/formatted/")[1])
        
#         st = pd.DataFrame({})
#         annot_sub = annotations[annotations['filename'] == os.path.basename(audio_fp)]
#         st["Begin Time (s)"] = annot_sub["onset"]
#         st["End Time (s)"] = annot_sub["offset"]
#         st["Annotation"] = annot_sub["event_label"]
        
#         new_st_fn = os.path.basename(audio_fp).replace('.wav', '.txt')
#         new_st_fp = os.path.join(annot_tgt_dir, new_st_fn)
        
#         st.to_csv(new_st_fp, sep='\t', index=False)
#         manifest['selection_table_fp'].append(new_st_fp.split("/formatted/")[1])
        
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
