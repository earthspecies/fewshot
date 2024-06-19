from fewshot.models.audio_plus_labelencoder.apply import infer

def main():  
    model_path='weights/atst-finetuned-40s-support-windowed.pt'
    atst_model_path='weights/atstframe_base.ckpt'
    support_audio_fp='example/support_audio.wav'
    support_selection_table_fp='example/support_selections.txt'
    query_audio_fp='example/query_audio.wav'
    
    out=infer(model_path,
              atst_model_path,
              support_audio_fp, 
              support_selection_table_fp, 
              query_audio_fp, 
              focal_annotation_label="POS",
              window_inference_query=True,
              window_inference_support=True,
              inference_normalize_rms=0.005,
              inference_threshold=None,
              inference_n_chunks_to_keep=5,
              inference_chunk_size_sec=8,
              support_duration_limit_sec=None
             )
        
    out.to_csv('example/predictions.txt', sep='\t', index=False)

if __name__ == "__main__":
    main()
