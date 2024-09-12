import os
from pathlib import Path
import torchaudio
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Define dataset paths
base_dir = "/home/davidrobinson/fewshot_data/data_large/"
dataset_name = "animalspeak"

input_dirs = [f"{dataset_name}_pseudovox", f"{dataset_name}_audio_trimmed"]
output_dirs = [f"{dataset_name}_pseudovox_16k", f"{dataset_name}_audio_trimmed_16k"]

# Create output directories if they don't exist
for output_dir in output_dirs:
    os.makedirs(os.path.join(base_dir, output_dir), exist_ok=True)

# Function to resample audio
def resample_file(input_file, output_file, target_sr=16000):
    try:
        # Load audio file
        waveform, orig_sr = torchaudio.load(input_file)
        
        # Resample if the original sample rate is different from the target
        if orig_sr != target_sr:
            resample_transform = torchaudio.transforms.Resample(orig_sr, target_sr)
            waveform = resample_transform(waveform)
        
        # Save resampled audio to output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure subdirectory exists
        torchaudio.save(output_file, waveform, target_sr)
        return f"Resampled {os.path.basename(input_file)}"
    except Exception as e:
        return f"Failed to resample {os.path.basename(input_file)}: {e}"

# Function to process directory
def process_directory(input_dir, output_dir):
    input_path = Path(base_dir) / input_dir
    output_path = Path(base_dir) / output_dir
    
    # Get all .wav files from the input directory
    input_files = list(input_path.rglob('*.wav'))
    
    # Prepare list of output file paths
    output_files = [output_path / file.relative_to(input_path) for file in input_files]

    # Use ProcessPoolExecutor to resample files in parallel
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(resample_file, input_file, output_file): input_file for input_file, output_file in zip(input_files, output_files)}

        # Use tqdm to show progress and process futures as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Resampling {input_dir}"):
            # print(future.result())  # Print the result of the completed future
            pass

# Process both pseudovox and trimmed directories
process_directory(input_dirs[0], output_dirs[0])
print("Resampling pseudovoxes complete")

process_directory(input_dirs[1], output_dirs[1])
print("Resampling background files complete")
