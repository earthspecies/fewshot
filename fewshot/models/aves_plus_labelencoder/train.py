import yaml
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import tqdm
from functools import partial
import os
from einops import rearrange
import yaml
from pathlib import Path
from torchvision.ops import sigmoid_focal_loss
import random
import bitsandbytes as bnb
import wandb

from fewshot.models.aves_plus_labelencoder.model import FewShotModel
from fewshot.models.aves_plus_labelencoder.params import parse_args, save_params
from fewshot.data.data import get_dataloader
from fewshot.models.aves_plus_labelencoder.inference import inference_dcase
from fewshot.dcase_evaluation.evaluation import evaluate as evaluate_dcase

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    import warnings
    warnings.warn("Only using CPU! Check CUDA")
    
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def get_optimizer(model, args):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    param_groups = [{'params': regularized, "weight_decay": 0.01}, {'params': not_regularized, 'weight_decay': 0.}]
    args.weight_decay = 0.01
    optimizer = bnb.optim.AdamW8bit(param_groups, lr=args.lr, amsgrad=True)
    return optimizer

def main(args):
    ## Setup
    args = parse_args(args)
    set_seed(args.seed)

    experiment_dir = os.path.join(args.project_dir, args.name)
    setattr(args, 'experiment_dir', str(experiment_dir))
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    if args.wandb:
        wandb.init(project="fewshot")
        wandb.config.update(args)

    save_params(args)
    
    model=FewShotModel(args)
    
    if args.previous_checkpoint_fp is not None:
        print(f"loading model weights from {args.previous_checkpoint_fp}")
        cp = torch.load(args.previous_checkpoint_fp)
        model.load_state_dict(cp["model_state_dict"])
    
    ## Training
    if args.n_train_steps>0:
        model = train(model, args) 
    
    print("Training Complete!")
    
    ## Evaluation
    evaluation_manifest=pd.read_csv(args.dcase_evaluation_manifest_fp)
    outputs = []
    print("Evaluation")
    for i, row in tqdm.tqdm(evaluation_manifest.iterrows(), total=len(evaluation_manifest)):
        # if '/RD/' in row['audio_fp']:
        #     continue
        d = inference_dcase(model, args, row['audio_fp'], row['annotation_fp'])
        outputs.append(d)
        
    outputs=pd.concat(outputs)
    output_fp = os.path.join(args.experiment_dir, "dcase_predictions.csv")
    outputs.to_csv(output_fp, index=False)
    
    evaluate_dcase(output_fp, args.dcase_ref_files_path, args.name, "DCASE", args.experiment_dir)

def get_loss_fn(args):
    def loss_fn(logits, query_labels, query_audio_denoised_spec, query_audio_denoised_spec_prediction):
        # query labels: 0: NEG, 1: UNK, 2: POS (currently 1 is not used)
        # query_audio_denoised_spec(_prediction) : (batch, channels, time)
        
        denoise_loss = (query_audio_denoised_spec - query_audio_denoised_spec_prediction) ** 2
        denoise_loss = torch.nan_to_num(denoise_loss) # rarely spec has nans? dunno why, mask it out
        denoise_loss = torch.mean(denoise_loss, 1) # (batch, time)
        denoise_loss_mask = query_labels>0
        denoise_loss = denoise_loss * denoise_loss_mask # remove sections with no focal pseudovox
        denoise_loss = torch.mean(denoise_loss)
        
        logits=torch.flatten(logits)
        query_labels=torch.flatten(query_labels)
        
        query_binary = torch.minimum(query_labels, torch.ones_like(query_labels)) #2->1
        detection_loss = sigmoid_focal_loss(logits, query_binary) #TODO add in loss hyperparams?
        unknown_mask = query_labels != 1
        detection_loss = detection_loss * unknown_mask # set loss of unknowns to 0
        detection_loss = torch.mean(detection_loss)
        return detection_loss, args.denoising_loss_weight*denoise_loss
        
    return loss_fn

def compute_metrics(logits, query_labels):
    logits=torch.flatten(logits)
    query_labels=torch.flatten(query_labels)
    mask = query_labels!=1
    
    logits_masked=logits[mask]
    query_labels_masked=query_labels[mask]
    
    pred_pos = logits_masked>=0.
    pred_neg = logits_masked<0.
    query_labels_pos = query_labels_masked>=0.5
    query_labels_neg = query_labels_masked<0.5
    
    TP = torch.sum(pred_pos * query_labels_pos)
    TN = torch.sum(pred_neg * query_labels_neg)
    FP = torch.sum(pred_pos * query_labels_neg)
    FN = torch.sum(pred_neg * query_labels_pos)
    
    acc = (TP+TN)/(TP+TN+FP+FN)
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
        
    return torch.nan_to_num(acc), torch.nan_to_num(prec), torch.nan_to_num(rec)
    

def train(model, args):
    model = model.to(device)
    model.train()
    torch.compile(model)
  
    loss_fn = get_loss_fn(args)
    melspec = torchaudio.transforms.MelSpectrogram(n_fft=800, n_mels=128, hop_length=320) #default n_mels = 128
    melspec = melspec.to(device)
  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad = True)
    # optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr, amsgrad = True)
    # optimizer = get_optimizer(model, args)
    
    if args.unfreeze_encoder_step>0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.unfreeze_encoder_step)
        warmup2_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.n_steps_warmup)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_train_steps, eta_min=0, last_epoch=- 1)
        scheduler=torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, warmup2_scheduler, cosine_scheduler], [args.unfreeze_encoder_step, args.n_steps_warmup+args.unfreeze_encoder_step], last_epoch=-1)
    else:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.n_steps_warmup)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_train_steps, eta_min=0, last_epoch=- 1)
        scheduler=torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], [args.n_steps_warmup], last_epoch=-1)
  
    # scaler = torch.cuda.amp.GradScaler()
    
    history={'detection_loss' : [], 'denoising_loss' : [], 'learning_rate' : [], 'accuracy' : [], 'precision':[], 'recall':[]}
    
    dataloader=get_dataloader(args)
    
    model.freeze_audio_encoder()
    
    for t, data_item in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if t == args.unfreeze_encoder_step:
            model.unfreeze_audio_encoder()
        
        support_audio, support_labels, query_audio, query_labels, query_audio_denoised = data_item
        
        # with torch.cuda.amp.autocast(dtype=torch.float16):
        logits, query_labels, query_audio_denoised_spec_prediction = model(support_audio.to(device = device, dtype = torch.float), support_labels.to(device = device, dtype = torch.float), query_audio.to(device = device, dtype = torch.float), query_labels=query_labels.to(device = device, dtype = torch.float))


        normalization_factor = torch.std(support_audio, dim=1, keepdim=True)
        normalization_factor = torch.maximum(normalization_factor, torch.full_like(normalization_factor, 1e-6))
        query_audio_denoised = (query_audio_denoised - torch.mean(query_audio_denoised, dim=1,keepdim=True)) / normalization_factor

        query_audio_denoised_spec = melspec(query_audio_denoised.to(device))[:,:,:query_labels.size(1)] # [batch, channels, time]. Need to chop off one extra time bin.
        query_audio_denoised_spec = torch.log(query_audio_denoised_spec + torch.full_like(query_audio_denoised_spec, 1e-6))
        query_audio_denoised_spec = query_audio_denoised_spec - torch.amin(query_audio_denoised_spec, (1,2), keepdim=True)
        query_audio_denoised_spec = query_audio_denoised_spec / (torch.amax(query_audio_denoised_spec, (1,2), keepdim=True) + torch.full_like(query_audio_denoised_spec, 1e-6))

        detection_loss, denoising_loss = loss_fn(logits, query_labels, query_audio_denoised_spec, query_audio_denoised_spec_prediction)
            
        acc, prec, rec = compute_metrics(logits, query_labels)
                
        history['detection_loss'].append(detection_loss.item())
        history['denoising_loss'].append(denoising_loss.item())
        history['learning_rate'].append(optimizer.param_groups[0]["lr"])
        history['accuracy'].append(float(acc.item()))
        history['precision'].append(float(prec.item()))
        history['recall'].append(float(rec.item()))

        if args.wandb:
            wandb.log(
                {
                    "detection_loss": detection_loss.item(),
                    "denoising_loss": denoising_loss.item(),
                    "accuracy": acc.item(),
                    "precision": prec.item(),
                    "recall": rec.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        # if t % args.log_steps == 0:
        #     print(f"Step {t}: Loss={np.mean(history['loss'][-10:])}, Accuracy={np.mean(history['accuracy'][-10:])}, Precision={np.mean(history['precision'][-10:])}, Recall={np.mean(history['recall'][-10:])}")


        # Backpropagation
        optimizer.zero_grad()
        loss = detection_loss+denoising_loss
        loss.backward()
        
        # scaler.scale(detection_loss+denoising_loss).backward()
        # scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        # scaler.step(optimizer)
        optimizer.step()
        scheduler.step()
        # scaler.update()
        
        if (t % args.checkpoint_frequency == 0) or (t==len(dataloader)-1):
            print(f"Step {t}: Detection Loss={np.mean(history['detection_loss'][-args.checkpoint_frequency:])}, Denoising Loss={np.mean(history['denoising_loss'][-args.checkpoint_frequency:])}, Accuracy={np.mean(history['accuracy'][-args.checkpoint_frequency:])}, Precision={np.mean(history['precision'][-args.checkpoint_frequency:])}, Recall={np.mean(history['recall'][-args.checkpoint_frequency:])}")
            with open(os.path.join(args.experiment_dir, "history.yaml"), 'w') as f:
                yaml.dump(history, f)
            checkpoint_dict = {
                "step": t,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                }

            torch.save(checkpoint_dict, os.path.join(args.experiment_dir, f"model_{t}.pt"))
            Path(os.path.join(args.experiment_dir, f"model_{int(t-3*args.checkpoint_frequency)}.pt")).unlink(missing_ok=True)
            
    return model

if __name__ == "__main__":
    main(sys.argv[1:])
