'''
Script to process original DCASE2022 or 2024 Task 5 data, so that training data can be used for few shot validation on-the-fly
'''

import os
import numpy as np
import soundfile as sf
import pandas as pd
from glob import glob
import shutil
import librosa
from tqdm import tqdm
import yaml

def main():
    DEV_SET_DIR='/home/jupyter/data/fewshot_data/evaluation/raw/DCASE_2024/Development_Set'
    target_dir = '/home/jupyter/data/fewshot_data/validation/formatted'
    
    n_clips_per_dataset = 100
    support_dur_sec = 60
    query_dur_sec = 30
    chunk_size_sec = 10
    n_vox_in_support = 5
    
    os.makedirs(target_dir, exist_ok=True)
    
    train_dir = os.path.join(DEV_SET_DIR, "Training_Set")
    val_dir = os.path.join(DEV_SET_DIR, "Validation_Set")
    
    for dname in ["BV", "HT", "JD", "MT", "WMW", "HB", "ME", "PB", "PB24", "PW", "RD"]:
        rng = np.random.default_rng(0)
        
        print(f"Processing {dname}")
        dtgt_dir = os.path.join(target_dir, dname)
        audio_tgt_dir = os.path.join(dtgt_dir, "audio")
        annot_tgt_dir = os.path.join(dtgt_dir, "selection_tables")
        
        os.makedirs(dtgt_dir, exist_ok=True)
        os.makedirs(audio_tgt_dir, exist_ok=True)
        os.makedirs(annot_tgt_dir, exist_ok=True)
        
        manifest = {"audio_fp": [], "selection_table_fp" : []}
        
        split = "val" if dname in ["HB", "ME", "PB", "PB24", "PW", "RD"] else "train"
        parent_dir = val_dir if split=="val" else train_dir
        
        audio_fps = sorted(glob(os.path.join(parent_dir, dname, "*.wav")))
        for x in audio_fps:
            st_fp = x.replace(".wav", ".csv")
            print(f"checking {st_fp}")
            assert os.path.exists(st_fp)
        
        for clipnumber in tqdm(range(n_clips_per_dataset)):
            ok_choice = False
            
            while not ok_choice:
                orig_audio_fp = rng.choice(audio_fps)
                orig_st_fp = orig_audio_fp.replace(".wav", ".csv")
                orig_st = pd.read_csv(orig_st_fp).sort_values("Starttime").reset_index(drop=True)

                audio_dur_sec = librosa.get_duration(path = orig_audio_fp)
                
                all_columns = orig_st.columns
                possible_columns = []
                for cc in all_columns:
                    if len(orig_st[orig_st[cc] == "POS"]) > n_vox_in_support:
                        possible_columns.append(cc)
                        
                if len(possible_columns) > 0:
                    ok_choice = True
                
            column = rng.choice(possible_columns)
            
            # Choose support audio interval
            
            allowed_start_idxs = orig_st[orig_st[column] == "POS"][:-n_vox_in_support].index
            assert len(allowed_start_idxs) > 0
            
            start_idx = rng.choice(allowed_start_idxs)
            idx = start_idx
            
            n_vox_included = 0
            while n_vox_included < n_vox_in_support:
                if orig_st.loc[idx, column] == "POS":
                    n_vox_included += 1
                idx += 1
                
            end_idx = idx
            
            st_for_support = orig_st.iloc[start_idx:end_idx]
            st_before_support = orig_st.iloc[:start_idx]
            st_after_support = orig_st.iloc[end_idx:]
            
            previous_vox_time = 0 if len(st_before_support) == 0 else min(st_for_support["Starttime"].min(), st_before_support["Endtime"].max())
            time_start_support = 0.5 * (st_for_support["Starttime"].min()+previous_vox_time)
            
            next_vox_time = audio_dur_sec if len(st_after_support) == 0 else max(st_after_support["Starttime"].min(), st_for_support["Endtime"].max())
            time_end_support = 0.5 * (st_for_support["Endtime"].max()+next_vox_time)
            
            support_audio, sr = librosa.load(orig_audio_fp, sr=None, offset = time_start_support, duration = time_end_support - time_start_support, mono=True)
            support_anno = np.zeros_like(support_audio)
                
            for i, row in st_for_support.iterrows():
                if row[column] == "POS":
                    start_sample = int((row["Starttime"]-time_start_support) * sr)
                    end_sample = int((row["Endtime"]-time_start_support) * sr)
                    support_anno[start_sample:end_sample] = 1
                elif row[column] == "UNK":
                    start_sample = int((row["Starttime"]-time_start_support) * sr)
                    end_sample = int((row["Endtime"]-time_start_support) * sr)
                    support_anno[start_sample:end_sample] = -1
                    
            # Subselect support audio down to 60 sec
            
            chunk_size_samples = int(sr * chunk_size_sec)
            n_chunks_to_keep = support_dur_sec // chunk_size_sec

            chunks_to_keep = []
            chunks_to_maybe_keep = []
            
            for chunk_start in np.arange(0, len(support_audio), chunk_size_samples):
                chunk_end = min(chunk_start+chunk_size_samples, len(support_audio))
                annot_chunk = support_anno[chunk_start:chunk_end]
                
                if np.amax(annot_chunk) >0:
                    chunks_to_keep.append(int(chunk_start))
                else:
                    chunks_to_maybe_keep.append(int(chunk_start))
                    
            chunks_to_keep = rng.permutation(chunks_to_keep)[:n_chunks_to_keep]
            n_remaining = n_chunks_to_keep - len(chunks_to_keep)
            if n_remaining > 0:
                more_chunks_to_keep = rng.permutation(chunks_to_maybe_keep)[:n_remaining]
                chunks_to_keep = np.concatenate([chunks_to_keep, more_chunks_to_keep]).astype(int)
                
            chunks_to_keep = sorted(chunks_to_keep)
            
            support_audio_final = []
            support_anno_final = []
            for c in chunks_to_keep:
                support_audio_final.append(support_audio[c:c+chunk_size_samples])
                support_anno_final.append(support_anno[c:c+chunk_size_samples])
            support_audio_final=np.concatenate(support_audio_final)
            support_anno_final=np.concatenate(support_anno_final)
            
            support_audio_final = support_audio_final[support_anno_final >=0]
            support_anno_final = support_anno_final[support_anno_final >=0]
            
            support_dur_samples = int(support_dur_sec * sr)
            if len(support_audio_final) < support_dur_samples:
                support_audio_final = np.concatenate([support_audio_final, np.zeros((support_dur_samples-len(support_audio_final),))])
                support_anno_final = np.concatenate([support_anno_final, np.zeros((support_dur_samples-len(support_anno_final),))])
                
            if not len(support_audio_final) == support_dur_samples:
                import pdb; pdb.set_trace()
            assert len(support_anno_final) == support_dur_samples
            
            # Choose query audio
            before_support = rng.binomial(1, 0.5)
            if before_support:
                query_st = st_before_support
                if len(query_st) == 0:
                    query_start_sec = rng.uniform(0, time_start_support)
                    
                else:
                    focal_idx = rng.choice(query_st.index)
                    focal_time = query_st.loc[focal_idx]["Starttime"]
                    query_start_sec = max(focal_time - rng.uniform(0, query_dur_sec/2), 0)
                    
                query_end_sec = min(query_start_sec + query_dur_sec, time_start_support)
             
            else:
                query_st = st_after_support
                if len(query_st) == 0:
                    query_start_sec = rng.uniform(time_end_support, audio_dur_sec)
                    
                else:
                    focal_idx = rng.choice(query_st.index)
                    focal_time = query_st.loc[focal_idx]["Starttime"]
                    query_start_sec = max(focal_time - rng.uniform(0, query_dur_sec/2), time_end_support)
                    
                query_end_sec = min(query_start_sec + query_dur_sec, audio_dur_sec)
                
            query_st = query_st[(query_st["Starttime"] >= query_start_sec) & (query_st["Starttime"] <= query_end_sec)]
            
            query_audio, sr = librosa.load(orig_audio_fp, sr=None, offset = query_start_sec, duration = query_end_sec-query_start_sec, mono=True)
            query_anno = np.zeros_like(query_audio)
            
            for i, row in query_st.iterrows():
                if row[column] == "POS":
                    start_sample = int((row["Starttime"]-query_start_sec) * sr)
                    end_sample = int((row["Endtime"]-query_start_sec) * sr)
                    query_anno[start_sample:end_sample] = 1
                elif row[column] == "UNK":
                    start_sample = int((row["Starttime"]-query_start_sec) * sr)
                    end_sample = int((row["Endtime"]-query_start_sec) * sr)
                    query_anno[start_sample:end_sample] = -1
            
            query_dur_samples = int(query_dur_sec * sr)
            
            if len(query_audio) < query_dur_samples:
                query_audio = np.concatenate([query_audio, np.zeros((query_dur_samples-len(query_audio),))])
                query_anno = np.concatenate([query_anno, np.zeros((query_dur_samples-len(query_anno),))])
                
            assert len(query_audio) == query_dur_samples
            assert len(query_anno) == query_dur_samples
            
            # Package into final form
            audio = np.concatenate([support_audio_final, query_audio])
            label_pos=(np.concatenate([support_anno_final, query_anno])==1)
            label_unk = (np.concatenate([support_anno_final, query_anno])==-1)
            
            d = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : []}
            
            starts = label_pos[1:] * ~label_pos[:-1]
            starts = np.where(starts)[0] + 1

            for start in starts:
                look_forward = label_pos[start:]
                ends = np.where(~look_forward)[0]
                if len(ends)>0:
                    end = start+np.amin(ends)
                else:
                    end = len(label_pos)-1
                d["Begin Time (s)"].append(start/sr)
                d["End Time (s)"].append(end/sr)
                d["Annotation"].append("POS")

            if label_pos[0]:
                start = 0
                look_forward = label_pos[start:]
                ends = np.where(~look_forward)[0]
                if len(ends)>0:
                    end = start+np.amin(ends)
                else:
                    end = len(label_pos)-1
                d["Begin Time (s)"].append(start/sr)
                d["End Time (s)"].append(end/sr)
                d["Annotation"].append("POS")
                
            starts_unk = label_unk[1:] * ~label_unk[:-1]
            starts_unk = np.where(starts_unk)[0] + 1

            for start in starts_unk:
                look_forward = label_unk[start:]
                ends = np.where(~look_forward)[0]
                if len(ends)>0:
                    end = start+np.amin(ends)
                else:
                    end = len(label_unk)-1
                d["Begin Time (s)"].append(start/sr)
                d["End Time (s)"].append(end/sr)
                d["Annotation"].append("UNK")

            if label_unk[0]:
                start = 0
                look_forward = label_unk[start:]
                ends = np.where(~look_forward)[0]
                if len(ends)>0:
                    end = start+np.amin(ends)
                else:
                    end = len(label_unk)-1
                d["Begin Time (s)"].append(start/sr)
                d["End Time (s)"].append(end/sr)
                d["Annotation"].append("UNK")
    
            d = pd.DataFrame(d)
            
            out_fp = os.path.join(annot_tgt_dir, f"{clipnumber}.txt")
            d.to_csv(out_fp, sep="\t", index=False)
            manifest["audio_fp"].append(out_fp.split(f"/{dname}/")[1])
            
            out_fp = os.path.join(audio_tgt_dir, f"{clipnumber}.wav")
            sf.write(out_fp, audio, sr)
            
            manifest["selection_table_fp"].append(out_fp.split(f"/{dname}/")[1])
            
        manifest = pd.DataFrame(manifest)
        manifest.to_csv(os.path.join(dtgt_dir, "manifest.csv"), index=False)
        
        metadata = {"n_clips_per_dataset" : n_clips_per_dataset, "support_dur_sec" : support_dur_sec, "query_dur_sec" : query_dur_sec, "chunk_size_sec" : chunk_size_sec, "n_vox_in_support" : n_vox_in_support}
        metadata_fp = os.path.join(dtgt_dir, "metadata.yaml")
        with open(metadata_fp, "w") as f:
            yaml.dump(metadata, f)

if __name__ == "__main__":
    main()
