import pandas as pd
import os
from glob import glob

dcase_data_parent_dir = "/home/jupyter/fewshot/data/DCASE2024_Development_Set/"

# val only
dcase_data_dir = os.path.join(dcase_data_parent_dir, "Development_Set") # use the versions copied into development set, since HB had files renamed
manifest = {'audio_fp': [], 'annotation_fp': []}

for dname in ["HB", "ME", "PB", "PB24", "PW", "RD"]:
    data_dir = os.path.join(dcase_data_dir, dname)
    audio_fps = sorted(glob(os.path.join(data_dir, "*.wav")))
    annot_fps = [x.replace(".wav", ".csv") for x in audio_fps]
    for x in annot_fps:
        assert os.path.exists(x)
    manifest['audio_fp'].extend(audio_fps)
    manifest['annotation_fp'].extend(annot_fps)
    
manifest=pd.DataFrame(manifest)
manifest.to_csv(os.path.join(dcase_data_parent_dir, "Validation_Set_manifest.csv"), index=False)

# all dev
dcase_data_dir = os.path.join(dcase_data_parent_dir, "Development_Set")
manifest = {'audio_fp': [], 'annotation_fp': []}

for dname in ["HB", "ME", "PB", "PB24", "PW", "RD", "BV", "HT", "JD", "MT", "WMW"]:
    data_dir = os.path.join(dcase_data_dir, dname)
    audio_fps = sorted(glob(os.path.join(data_dir, "*.wav")))
    annot_fps = [x.replace(".wav", ".csv") for x in audio_fps]
    for x in annot_fps:
        assert os.path.exists(x)
    manifest['audio_fp'].extend(audio_fps)
    manifest['annotation_fp'].extend(annot_fps)
    
manifest=pd.DataFrame(manifest)
manifest.to_csv(os.path.join(dcase_data_parent_dir, "Development_Set_manifest.csv"), index=False)
    
    