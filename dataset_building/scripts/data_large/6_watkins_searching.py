import os
import shutil
from pydub import AudioSegment

# Define the input and output paths
input_folder = "/home/ubuntu/fewshot_data/data_large/watkins_pseudovox2"
output_folder = "/home/ubuntu/fewshot_data/data_large/watkins_pseudovox2_long"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize variables
processed_files = 0
max_files = 50
min_duration_ms = 2000  # 2 seconds in milliseconds

# Supported audio file extensions
audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')

# Iterate over files in the input folder
for filename in os.listdir(input_folder):
    if processed_files >= max_files:
        break

    # Construct full file path
    file_path = os.path.join(input_folder, filename)

    # Check if it's a file and has an audio extension
    if os.path.isfile(file_path) and filename.lower().endswith(audio_extensions):
        try:
            # Load audio file
            audio = AudioSegment.from_file(file_path)
            duration = len(audio)  # Duration in milliseconds

            # Check if duration is longer than 2 seconds
            if duration > min_duration_ms:
                # Copy file to output folder
                shutil.copy(file_path, output_folder)
                processed_files += 1
                print(f"Copied {filename} to {output_folder}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"Total files processed: {processed_files}")
