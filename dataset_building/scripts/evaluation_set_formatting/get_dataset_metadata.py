import os
import subprocess
import librosa
import pandas as pd
from tqdm import tqdm

# Define the LaTeX table content

datasets = ["marmoset", "Anuraset", "carrion_crow", "katydid", "Spiders", "rana_sierrae", "Powdermill",  "Hawaii", "right_whale", "gibbons", "gunshots", "humpback", "ruffed_grouse", "audioset_strong", "DESED"]
dataset_names = [x.replace('_', ' ').title() for x in datasets]
formatted_dataset_parent_dir = "/home/jupyter/data/fewshot_data/evaluation/formatted"

columns = ["Dataset", "N files", "Duration (hr)", "N events", "Recording type", "Taxa"] #, "Location"]

tex = r"""
\documentclass{article}
\usepackage{booktabs}
\begin{document}

\begin{table}[h]
    \centering
    \begin{tabular}{||"""

tex+= f"{len(columns)*'c|'}"

tex += r"""|}
        \toprule
        
"""

for i, column in enumerate(columns):
    tex += f"{column}"
    if i < len(columns)-1:
        tex += " & "
    else:
        tex += r""" \\
        \midrule
        
        """
        
for dataset_name, dataset in sorted(zip(dataset_names, datasets)):
    data_dir = os.path.join(formatted_dataset_parent_dir, dataset)
    manifest = pd.read_csv(os.path.join(data_dir, "manifest.csv"))
    
    total_dur = 0
    n_events = 0
    
    for i, row in tqdm(manifest.iterrows(), total=len(manifest)):
        audio_fp = os.path.join(formatted_dataset_parent_dir, row["audio_fp"])
        total_dur += librosa.get_duration(path=audio_fp, sr=None) / 3600
        
        st_fp = os.path.join(formatted_dataset_parent_dir, row["selection_table_fp"])
        st = pd.read_csv(st_fp, sep='\t')
        st = st[st["Annotation"] != "Unknown"]
        n_events += len(st)
    
    for i, column in enumerate(columns):
        if column == "Dataset":
            entry = dataset_name
        elif column == "N files":
            entry = len(manifest)
        elif column == "Duration (hr)":
            entry = '%.2f'%(total_dur)
        elif column == "N events":
            entry = n_events
        elif column == "Recording type":
            dd = {"marmoset" : "Laboratory", "Anuraset" : "Terrestrial PAM", "carrion_crow" : "On-body", "katydid" : "Terrestrial PAM", "Spiders" : "Substrate", "rana_sierrae" : "Underwater PAM", "Powdermill" : "Terrestrial PAM",  "Hawaii" : "Terrestrial PAM", "right_whale" : "Underwater PAM", "gibbons" : "Terrestrial PAM", "gunshots" : "Terrestrial PAM", "humpback" : "Underwater PAM", "ruffed_grouse" : "Terrestrial PAM", "audioset_strong" : "Various", "DESED" : "Various"}
            
            entry = dd[dataset]
        elif column == "Taxa":
            dd = {"marmoset" : "Callithrix jacchus", "Anuraset" : "Anura", "carrion_crow" : "Corvus corone", "katydid" : "Tettigoniidae", "Spiders" : "Salticidae (?)", "rana_sierrae" : "Rana sierrae", "Powdermill" : "Passeriformes",  "Hawaii" : "Aves", "right_whale" : "Eubalaena glacialis", "gibbons" : "Nomascus hainanus", "gunshots" : "Homo sapiens", "humpback" : "Megaptera novaeangliae", "ruffed_grouse" : "Bonasa umbellus", "audioset_strong" : "n/a", "DESED" : "n/a"}
            
            entry = dd[dataset]
            
        elif column == "Location":
#             dd = {"marmoset" : "Callithrix jacchus", "Anuraset" : "Anura", "carrion_crow" : "Corvus corone", "katydid" : "Tettigoniidae", "Spiders" : "Salticidae (?)", "rana_sierrae" : "Rana sierrae", "Powdermill" : "Aves",  "Hawaii" : "Aves", "right_whale" : "Eubalaena glacialis", "gibbons" : "Nomascus hainanus", "gunshots" : "Homo sapiens", "humpback" : "Megaptera novaeangliae", "ruffed_grouse" : "Bonasa umbellus"}
            
#             entry = dd[dataset]
            entry = " "
        else:
            entry = " "
            
        tex += f"{entry}"
        if i < len(columns)-1:
            tex += " & "
        else:
            tex += r""" \\
            \midrule

            """
    
    
#     print(f"Processing {dataset}")
    
#     
    
#     

        
tex += r"""
        \bottomrule
    \end{tabular}
    \caption{Sample Table}
    \label{tab:sample}
\end{table}
\end{document}
"""

target_directory = formatted_dataset_parent_dir

# Ensure the target directory exists
os.makedirs(target_directory, exist_ok=True)

# Define the LaTeX source file path and PDF file path
latex_file_path = os.path.join(target_directory, 'metadata.tex')
pdf_file_path = os.path.join(target_directory, 'metadata.pdf')

# Write the LaTeX code to a file
with open(latex_file_path, 'w') as f:
    f.write(tex)

# Compile the LaTeX file to PDF
try:
    subprocess.run(['pdflatex', '-output-directory', target_directory, latex_file_path], check=True)
    print(f"PDF successfully created at {pdf_file_path}")
except subprocess.CalledProcessError as e:
    print(f"Error compiling LaTeX: {e}")
