#!/bin/bash

cd "/home/ubuntu/fewshot_data/data_large/"
dataset_name="watkins"

##########
# Check if the input folder exists
if [ ! -d "${dataset_name}_pseudovox2" ]; then
    echo "Error: Input folder 'pseudovox' not found"
    # exit 1
fi

# Create output folder if it doesn't exist
mkdir -p ${dataset_name}_pseudovox_16k2

# Define a function to resample a single file
resample_file() {
    dataset_name="watkins"
    input_file="${dataset_name}_pseudovox2/$(basename "$1")"
    output_file="${dataset_name}_pseudovox_16k2/$(basename "$input_file")"
    ffmpeg -i "$input_file" -ar 16000 "$output_file" >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Resampled $(basename "$input_file")"
    else
        echo "Failed to resample $(basename "$input_file")"
    fi
}

# Export the function so that parallel can use it
export -f resample_file

# Use GNU Parallel to process files in parallel and display progress with ETA
find ${dataset_name}_pseudovox2 -type f -name '*.wav' | parallel --eta resample_file {}

echo "Resampling pseudovoxes complete"

# ###########
# # Check if the input folder exists
# if [ ! -d "${dataset_name}_resampled2" ]; then
#     echo "Error: Input folder 'audio_trimmed' not found"
#     # exit 1
# fi

# # Create output folder if it doesn't exist
# mkdir -p ${dataset_name}_audio_trimmed_16k

# # Define a function to resample a single file
# resample_file() {
#     dataset_name="watkins"
#     input_file="${dataset_name}_resampled2/$(basename "$1")"
#     output_file="${dataset_name}_audio_trimmed_16k/$(basename "$input_file")"
#     ffmpeg -i "$input_file" -ar 16000 "$output_file" >/dev/null 2>&1
#     if [ $? -eq 0 ]; then
#         echo "Resampled $(basename "$input_file")"
#     else
#         echo "Failed to resample $(basename "$input_file")"
#     fi
# }

# # Export the function so that parallel can use it
# export -f resample_file

# # Use GNU Parallel to process files in parallel and display progress with ETA
# find ${dataset_name}_resampled -type f -name '*.wav' | parallel --progress resample_file {}

echo "Resampling background files complete"
