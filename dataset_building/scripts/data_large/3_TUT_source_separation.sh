mixit_path="/home/davidrobinson/sound-separation/"
info_path="/home/davidrobinson/fewshot_data/data_large/TUT_temp_mixit_manifest.csv"

cd ${mixit_path}
python3 ${mixit_path}models/tools/process_wav_stitching_batched.py \
    --model_dir ${mixit_path}bird_mixit_model_checkpoints/output_sources4 \
    --checkpoint ${mixit_path}bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090 \
    --block_size_in_seconds 10 --permutation_invariant True --window_type vorbis \
    --info ${info_path} \
    --sample_rate 22050
            
# rm ${info_path}

# cd ${mixit_path}
# python3 ${mixit_path}models/tools/process_wav_batched.py \
#         --model_dir ${mixit_path}/weights/bird_mixit_model_checkpoints/output_sources4 \
#         --checkpoint ${mixit_path}/weights/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090 \
#         --num_sources 4 \
#         --info ${info_path}
        
