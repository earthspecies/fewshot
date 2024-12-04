'''
Script to process original audioset_strong data, so that training data can be used for few shot evaluation
'''

import os
import pandas as pd
from glob import glob
import shutil
import soundfile as sf
import librosa
import numpy as np

def main():
    DATA_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/audioset_strong'
    target_dir = '/home/jupyter/data/fewshot_data/evaluation/formatted'
    rng = np.random.default_rng(0)
    
    os.makedirs(target_dir, exist_ok=True)
    
    dname = "audioset_strong"
    dtgt_dir = os.path.join(target_dir, dname)
    audio_tgt_dir = os.path.join(dtgt_dir, "audio")
    annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
    
    if os.path.exists(dtgt_dir):
        shutil.rmtree(dtgt_dir)
    
    os.makedirs(dtgt_dir, exist_ok=True)
    os.makedirs(audio_tgt_dir, exist_ok=True)
    os.makedirs(annot_tgt_dir, exist_ok=True)
    manifest = {"audio_fp": [], "selection_table_fp" : []}
    metadata = {'audio_fn' : [], 'audioset_ids' : [], 'label' : []}
    
    annotations = pd.read_csv(os.path.join(DATA_DIR, 'audioset_eval_strong.tsv'), sep='\t')
    
    annotations["Duration"] = annotations["end_time_seconds"] - annotations["start_time_seconds"]
    
    annotation_key = pd.read_csv(os.path.join(DATA_DIR, 'mid_to_display_name.tsv'), sep='\t', names=['mid', 'name'])
    annotation_dict = {}
    for i, row in annotation_key.iterrows():
        annotation_dict[row['mid']] = row['name']
    
    annotation_count = {"anno" : [], "count" : []}
    for annotation in annotations["label"].unique():
        
        annot_sub = annotations[annotations["Duration"] < 8]
        annot_sub = annot_sub[annot_sub["start_time_seconds"]>0]
        annot_sub = annot_sub[annot_sub["end_time_seconds"]<10]
        
        annot_sub = annot_sub[annot_sub["label"] == annotation]
        annotation_count["anno"].append(annotation)
        annotation_count["count"].append(len(annot_sub))
        
    annotation_count = pd.DataFrame(annotation_count)
    annotation_count = annotation_count[annotation_count["count"] > 15] # require at least 15 files per anno type
    
    valid_annos = sorted(rng.permutation(sorted(annotation_count["anno"]))[:100]) # use up to 100 labels, chosen randomly
    
    for anno in valid_annos:
        print(f"Processing {annotation_dict[anno]} files")
        annot_sub = annotations[annotations["Duration"] <8] # filter out some files to reduce overlapping boxes at boundaries
        annot_sub = annot_sub[annot_sub["start_time_seconds"]>0]
        annot_sub = annot_sub[annot_sub["end_time_seconds"]<10]
        
        annot_sub = annot_sub[annot_sub["label"] == anno]
        valid_files = annot_sub["segment_id"].unique()
        
        for i in range(5): # create 5 examples per annotation type
            files_to_use = rng.permutation(valid_files) # grab 6 files per example 
            
            audio = []
            st = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : [], "Original Filename" : []}
            shift = 0
            nfiles = 0
            
            files_used = []
            
            for file in files_to_use:
                
                audio_fp = sorted(glob(os.path.join(DATA_DIR, "eval_segments/**", "*" + '_'.join(file.split('_')[:-1])+"*.wav")))
                if len(audio_fp) == 0:
                    print("can't find file, continuing")
                    continue
                
                if len(audio_fp) != 1:
                    import pdb; pdb.set_trace()
                audio_fp = audio_fp[0]
                a, sr = librosa.load(audio_fp, sr=16000, mono=True)
                
                audio.append(a)
                
                selections = annotations[annotations['segment_id'] == file]
                selections = selections[selections['label'] == anno]
                st["Begin Time (s)"].extend(selections["start_time_seconds"] + shift)
                st["End Time (s)"].extend(selections["end_time_seconds"].map(lambda x : min(len(a)/sr, x)) + shift)
                st["Annotation"].extend(selections["label"].map(lambda x : annotation_dict[x]))
                st["Original Filename"].extend([file for _ in range(len(selections))])
                shift += len(a) / sr
                
                files_used.append(file)
                
                nfiles+=1
                if nfiles == 6:
                    break
                
            audio = np.concatenate(audio)
            st = pd.DataFrame(st)
            
            st = st[(st["End Time (s)"] - st["Begin Time (s)"])>0.01]
            
            new_audio_fn = f"{annotation_dict[anno]}_{i}.wav"
            new_anno_fn = f"{annotation_dict[anno]}_{i}.txt"
            
            new_audio_fp = os.path.join(audio_tgt_dir, new_audio_fn)
            new_st_fp = os.path.join(annot_tgt_dir, new_anno_fn)
            
            # merge overlaps
            st_annos = sorted(st["Annotation"].unique())

            d = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : []}

            for ann in st_annos:
                st_sub = st[st["Annotation"] == ann]

                dur_max_ann = st_sub["End Time (s)"].max() + 1
                rr = 16000
                ann_merged_np = np.zeros((int(rr * dur_max_ann),), dtype=bool)

                for i, row in st_sub.iterrows():
                    begin = int(row["Begin Time (s)"] * rr)
                    end = int(row["End Time (s)"] * rr)
                    ann_merged_np[begin:end] = True

                starts = ann_merged_np[1:] * ~ann_merged_np[:-1]
                starts = np.where(starts)[0] + 1

                for start in starts:
                    look_forward = ann_merged_np[start:]
                    ends = np.where(~look_forward)[0]
                    if len(ends)>0:
                        end = start+np.amin(ends)
                    else:
                        end = len(ann_merged_np)-1
                    d["Begin Time (s)"].append(start/rr)
                    d["End Time (s)"].append(end/rr)
                    d["Annotation"].append(ann)

                if ann_merged_np[0]:
                    start = 0
                    look_forward = ann_merged_np[start:]
                    ends = np.where(~look_forward)[0]
                    if len(ends)>0:
                        end = start+np.amin(ends)
                    else:
                        end = len(ann_merged_np)-1
                    d["Begin Time (s)"].append(start/rr)
                    d["End Time (s)"].append(end/rr)
                    d["Annotation"].append(ann)

            st = pd.DataFrame(d)
            # end merge overlaps
            
            sf.write(new_audio_fp, audio, sr)
            st.to_csv(new_st_fp, sep='\t', index=False)
            
            metadata['audio_fn'].append(os.path.basename(new_audio_fp))
            metadata['audioset_ids'].append(' '.join(files_used))
            metadata['label'].append(anno)
            
            manifest["audio_fp"].append(new_audio_fp.split("/formatted/")[1])
            manifest["selection_table_fp"].append(new_st_fp.split("/formatted/")[1])  
    
    pd.DataFrame(metadata).to_csv(os.path.join(dtgt_dir, "metadata.csv"), index=False)
    pd.DataFrame(manifest).to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)

if __name__ == "__main__":
    main()
