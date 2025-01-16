import os
import pandas as pd
import librosa
import soundfile as sf
import shutil
import numpy as np
from glob import glob

parent_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted"
voxaboxen_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted_for_voxaboxen"

datasets = ["katydid", "marmoset", "Spiders", "Anuraset", "carrion_crow", "rana_sierrae", "Powdermill",  "Hawaii", "right_whale", "gibbons", "gunshots", "humpback", "ruffed_grouse", "audioset_strong", "DESED"]
datasets += [f"{x}_crossfile" for x in ["katydid", "marmoset", "Spiders", "Anuraset", "carrion_crow", "rana_sierrae", "Powdermill",  "Hawaii", "right_whale", "gibbons", "gunshots", "humpback", "ruffed_grouse"]]
datasets += ["katydid_sixthtime", "katydid_crossfile_sixthtime"]
datasets += ["marmoset_halftime", "marmoset_crossfile_halftime"]
datasets += ["Spiders_halftime", "Spiders_crossfile_halftime"]
datasets += ["PB", "PB24", "HB", "PB_halftime", "PB24_halftime", "ME", "RD", "PW"]

def get_breakpoint(st, break_after = 5):
    st_sub = st[st["Annotation"] != "Unknown"]
    endtime = sorted(st_sub["End Time (s)"])[break_after-1]
    return endtime

commands_to_run = ""

for dataset in datasets:
    print(f"Processing {dataset}")
    dataset_dir = os.path.join(parent_dir, dataset)
    new_dataset_dir = os.path.join(voxaboxen_dir, dataset)
    
    if os.path.exists(new_dataset_dir):
        shutil.rmtree(new_dataset_dir)
    
    new_audio_dir = os.path.join(new_dataset_dir, "audio")
    new_st_dir = os.path.join(new_dataset_dir, "selection_tables")
    
    for d in [new_audio_dir, new_st_dir]:
        os.makedirs(d)
    
    manifest_fp = os.path.join(dataset_dir, "manifest.csv")
    manifest=pd.read_csv(manifest_fp)
    for i, row in manifest.iterrows():
        audio_fp = os.path.join(parent_dir, row["audio_fp"])
        
        st_fp = os.path.join(parent_dir, row["selection_table_fp"])
        st = pd.read_csv(st_fp, sep = '\t')
        breakpoint = get_breakpoint(st)
        
        audio, sr = librosa.load(audio_fp, duration = breakpoint, sr = None)
        
        support_audio_fp = audio_fp.replace(parent_dir, voxaboxen_dir)
        sf.write(support_audio_fp, audio, sr)
        
        support_st = st[st["Begin Time (s)"] < breakpoint].copy()
        support_st["End Time (s)"] = support_st["End Time (s)"].map(lambda x : min(x, breakpoint))
        
        support_st_fp = st_fp.replace(parent_dir, voxaboxen_dir)
        support_st.to_csv(support_st_fp, sep='\t', index=False)
        
        train_manifest = pd.DataFrame({"fn" : [os.path.splitext(os.path.basename(support_audio_fp))[0]], "audio_fp" : [support_audio_fp], "selection_table_fp" : [support_st_fp]})
        train_info_fp = os.path.join(new_dataset_dir, f"{os.path.basename(support_audio_fp)}_support_info.csv")
        train_manifest.to_csv(train_info_fp, index=False)
        
        test_manifest = pd.DataFrame({"fn" : [os.path.splitext(os.path.basename(audio_fp))[0]], "audio_fp" : [audio_fp], "selection_table_fp" : [st_fp]})
        test_info_fp = os.path.join(new_dataset_dir, f"query_{os.path.basename(support_audio_fp)}_query_info.csv")
        test_manifest.to_csv(test_info_fp, index= False)
        
        project_name = dataset+"___"+os.path.splitext(os.path.basename(audio_fp))[0]
        
        support_st["Duration"] = support_st["End Time (s)"] - support_st["Begin Time (s)"]
        support_st_sub = support_st[support_st["Annotation"] != "Unknown"]
        fill_holes_dur_sec = min(support_st_sub["Duration"].min() * 0.5, 1)
        delete_short_dur_sec = min(support_st_sub["Duration"].min() * 0.5, 0.5)
        
        if breakpoint < 5:
            clip_duration = 0.6
            batch_size = 1
        else:
            continue
        
        # elif breakpoint < 30:
        #     clip_duration = 2
        #     batch_size = 1
        # else:
        #     clip_duration = 4
        #     batch_size = 4
            
        
        project_command = f'python main.py project-setup --train-info-fp="{train_info_fp}" --val-info-fp="{test_info_fp}" --test-info-fp="{test_info_fp}" --project-dir="/home/jupyter/fewshot_supervised/{project_name}"\n'
        commands_to_run += project_command
        
        train_command = f'python main.py train-model --project-config-fp="/home/jupyter/fewshot_supervised/{project_name}/project_config.yaml" --lr=.01 --batch-size={batch_size} --name=e2v1 --clip-duration={clip_duration} --segmentation-based --encoder=beats --unfreeze-encoder-epoch=1000000 --n-epochs=100 --rho=1 --delete-short-dur-sec={delete_short_dur_sec} --fill-holes-dur-sec={fill_holes_dur_sec} --overwrite --patience=10 --omit-empty-clip-prob=0.5\n'
        
        commands_to_run += train_command
        
#         train_command = f'python main.py train-model --project-config-fp="/home/jupyter/fewshot_supervised/{project_name}/project_config.yaml" --lr=.001 --batch-size={batch_size} --name=e3 --clip-duration={clip_duration} --segmentation-based --encoder=beats --unfreeze-encoder-epoch=1000000 --n-epochs=100 --rho=1 --delete-short-dur-sec={delete_short_dur_sec} --fill-holes-dur-sec={fill_holes_dur_sec} --overwrite --patience=10 --omit-empty-clip-prob=0.5\n'
        
#         commands_to_run += train_command
        
#         train_command = f'python main.py train-model --project-config-fp="/home/jupyter/fewshot_supervised/{project_name}/project_config.yaml" --lr=.0001 --batch-size={batch_size} --name=e4 --clip-duration={clip_duration} --segmentation-based --encoder=beats --unfreeze-encoder-epoch=1000000 --n-epochs=100 --rho=1 --delete-short-dur-sec={delete_short_dur_sec} --fill-holes-dur-sec={fill_holes_dur_sec} --overwrite --patience=10 --omit-empty-clip-prob=0.5\n'
        
#         commands_to_run += train_command
        
with open('/home/jupyter/sound_event_detection/fewshot_supervised.sh', 'w') as f:
    f.write(commands_to_run)
        