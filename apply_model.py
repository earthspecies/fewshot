
def main():  
  
    from fewshot.models.aves_plus_labelencoder.apply import infer
    
    infer('/home/jupyter/fewshot/projects/aves_plus_labelencoder_med/cls_token_longtrain/model_99999.pt',
          '/home/jupyter/fewshot/projects/aves_plus_labelencoder_med/cls_token_longtrain/params.yaml', 
          '/home/jupyter/orangutan_pulses/Training Sounds - Focal OU Pulses/2015-03-15 055216_1105-OTTO_LC_AH.WAV', 
          '/home/jupyter/orangutan_pulses/Training Selections- Focal OU Pulses/2015-03-15 055216_1105-OTTO_LC_AH.txt', 
          '/home/jupyter/orangutan_pulses/Training Sounds - Focal OU Pulses/2015-03-15 052914_1104-OTTO_LC_AH.WAV', 
          out_fp='/home/jupyter/2015-03-15 052914_1104-OTTO_LC_AH_predictions.txt')

if __name__ == "__main__":
    main()
