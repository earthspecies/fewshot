mixit_path="/home/jupyter/sound-separation/"
info_path="/home/jupyter/fewshot/data/data_small/temp_mixit_manifest.csv"

cd ${mixit_path}
python3 ${mixit_path}models/tools/process_wav_batched.py \
        --model_dir ${mixit_path}/weights/bird_mixit_model_checkpoints/output_sources4 \
        --checkpoint ${mixit_path}/weights/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090 \
        --num_sources 4 \
        --info ${info_path}
        
rm ${info_path}
