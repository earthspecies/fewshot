import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Define the root directory containing the subdirectories with .zip files
root_dir = '../fewshot_data/data_large/watkins_mastertapes/'

# Function to unzip a single file
def unzip_file(zip_path, extract_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        return f"Extracted {os.path.basename(zip_path)} to {extract_path}"
    except Exception as e:
        return f"----failed---- to extract {os.path.basename(zip_path)} at {extract_path}: {e}"

# Collect all .zip files to be processed
zip_files = []
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.zip'):
            zip_path = os.path.join(subdir, file)
            extract_path = subdir
            zip_files.append((zip_path, extract_path))


# zip_files = zip_files[500:]

# Run unzipping in parallel using ThreadPoolExecutor with progress bar
with ProcessPoolExecutor(32) as executor:
    futures = [executor.submit(unzip_file, zip_path, extract_path) for zip_path, extract_path in zip_files]
    
    # Using tqdm to display progress
    for future in tqdm(as_completed(futures), total=len(futures), desc="Unzipping files"):
        result = future.result()
        print(result)
