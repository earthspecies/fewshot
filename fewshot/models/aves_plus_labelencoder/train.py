import yaml
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from functools import partial
from einops import rearrange
from pathlib import Path
from torchvision.ops import sigmoid_focal_loss
import random
import bitsandbytes as bnb
import wandb

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from fewshot.models.aves_plus_labelencoder.model import FewShotModel
from fewshot.models.aves_plus_labelencoder.params import parse_args, save_params
from fewshot.data.data import FewshotDataset, get_dataloader, get_dataloader_distributed
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

def main(args):
    ## Setup
    args = parse_args(args)
    set_seed(args.seed)

    experiment_dir = os.path.join(args.project_dir, args.name)
    setattr(args, 'experiment_dir', str(experiment_dir))
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    save_params(args)

    if args.train:
        world_size = torch.cuda.device_count()
        dataset = FewshotDataset(args)

        if world_size > 1:
            mp.spawn(train, args=(dataset, world_size, initialize_model, args), nprocs=world_size, join=True)
        else:
            train(0, dataset, world_size, initialize_model, args, single_gpu=True)
        
        print("Training Complete!")

        model = initialize_model(args)
        final_model_fp = os.path.join(args.experiment_dir, "final_model.pt")
        cp = torch.load(final_model_fp)
        model.load_state_dict(cp) #["model_state_dict"])
    else:
        model = initialize_model(args)
    
    ## Evaluation
    evaluation_manifest = pd.read_csv(args.dcase_evaluation_manifest_fp)
    outputs = []
    print("Evaluation")
    for i, row in tqdm.tqdm(evaluation_manifest.iterrows(), total=len(evaluation_manifest)):
        # if "ME1" not in row['audio_fp']: # quick
        #     continue
        d = inference_dcase(model, args, row['audio_fp'], row['annotation_fp'])
        outputs.append(d)
        
    outputs = pd.concat(outputs)
    
    output_fp = os.path.join(args.experiment_dir, "dcase_predictions.csv")
    outputs.to_csv(output_fp, index=False)
    
    evaluate_dcase(output_fp, args.dcase_ref_files_path, args.name, "DCASE", args.experiment_dir)

def get_loss_fn(args):
    def loss_fn(logits, query_labels):
        logits = torch.flatten(logits)
        query_labels = torch.flatten(query_labels)
        
        query_binary = torch.minimum(query_labels, torch.ones_like(query_labels))  # 2->1
        # loss = F.binary_cross_entropy_with_logits(logits, query_binary) 
        loss = sigmoid_focal_loss(logits, query_binary)  
        unknown_mask = query_labels != 1
        loss = loss * unknown_mask  # set loss of unknowns to 0
        loss = torch.mean(loss)
        return loss
        
    return loss_fn

def compute_metrics(logits, query_labels):
    logits = torch.flatten(logits)
    query_labels = torch.flatten(query_labels)
    mask = query_labels != 1
    
    logits_masked = logits[mask]
    query_labels_masked = query_labels[mask]
    
    pred_pos = logits_masked >= 0.
    pred_neg = logits_masked < 0.
    query_labels_pos = query_labels_masked >= 0.5
    query_labels_neg = query_labels_masked < 0.5
    
    TP = torch.sum(pred_pos * query_labels_pos)
    TN = torch.sum(pred_neg * query_labels_neg)
    FP = torch.sum(pred_pos * query_labels_neg)
    FN = torch.sum(pred_neg * query_labels_pos)
    
    acc = (TP + TN) / (TP + TN + FP + FN)
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
        
    return torch.nan_to_num(acc), torch.nan_to_num(prec), torch.nan_to_num(rec)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # You can choose any available port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def initialize_model(args):
    model = FewShotModel(args)
    if args.previous_checkpoint_fp is not None:
        print(f"loading model weights from {args.previous_checkpoint_fp}")
        cp = torch.load(args.previous_checkpoint_fp)
        model.load_state_dict(cp["model_state_dict"])
    return model

def get_optimizer_layerwise(model, args):
    # Separate parameters into groups for LLRD
    regularized = []
    not_regularized = []
    tfm_params = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]  # Adjust this based on the number of transformer layers

    for name, param in model.named_parameters():
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
        
        # Check for atst layer parameters and group them
        if "atst_frame" in name:
            if "blocks.0." in name:
                tfm_params[1].append(param)
            elif "blocks.1." in name:
                tfm_params[2].append(param)
            elif "blocks.2." in name:
                tfm_params[3].append(param)
            elif "blocks.3." in name:
                tfm_params[4].append(param)
            elif "blocks.4." in name:
                tfm_params[5].append(param)
            elif "blocks.5." in name:
                tfm_params[6].append(param)
            elif "blocks.6." in name:
                tfm_params[7].append(param)
            elif "blocks.7." in name:
                tfm_params[8].append(param)
            elif "blocks.8" in name:
                tfm_params[9].append(param)
            elif "blocks.9." in name:
                tfm_params[10].append(param)
            elif "blocks.10." in name:
                tfm_params[11].append(param)
            elif "blocks.11." in name:
                tfm_params[12].append(param)
            elif ".norm_frame." in name:
                tfm_params[13].append(param)
            else:
                tfm_params[0].append(param)
    
    # Layer-wise learning rate scaling
    init_lr = args.lr
    lr_scale = args.tfm_lr_scale  # Ensure this is defined in args
    trainable_layers = len(tfm_params)
    scale_lrs = [init_lr * (lr_scale ** i) for i in range(trainable_layers)]

    tfm_param_groups = [{"params": tfm_params[i], "lr": scale_lrs[i]} for i in range(trainable_layers) if tfm_params[i]]

    # Regular optimizer param groups
    param_groups = [
        {'params': regularized, "weight_decay": 0.01},
        {'params': not_regularized, 'weight_decay': 0.}
    ] # + tfm_param_groups

    optimizer = bnb.optim.AdamW8bit(param_groups, lr=args.lr, amsgrad=True)
    return optimizer


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

def train(rank, dataset, world_size, model_fn, args, single_gpu=False):
    print(f"Running on rank {rank}, dataset type {type(dataset)}")
    
    if not single_gpu:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (rank == 0) and args.wandb:
        wandb.init(project="fewshot")
        wandb.config.update(args)
    
    model = model_fn(args)
    model = model.to(device)
    model.freeze_audio_encoder()
    
    if not single_gpu:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    model.train()
    torch.compile(model, mode="max-autotune")
  
    loss_fn = get_loss_fn(args)
    optimizer = get_optimizer(model, args)

    if not single_gpu:
        dataloader = get_dataloader_distributed(dataset, args, world_size=world_size, rank=rank)
    else:
        dataloader = get_dataloader(args)
    
    assert args.n_steps_warmup % args.gradient_accumulation_steps == 0, "Warmup must be divisible by gradient accumulation steps to ensure proper scheduling"
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.n_steps_warmup//args.gradient_accumulation_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader), eta_min=0, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], [args.n_steps_warmup//args.gradient_accumulation_steps], last_epoch=-1)
    
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    history = {'loss': [], 'learning_rate': [], 'accuracy': [], 'precision': [], 'recall': []}
    
    for t, data_item in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if t == args.unfreeze_encoder_step:
            if not single_gpu:
                model.module.unfreeze_audio_encoder()
            else:
                model.unfreeze_audio_encoder()
        
        support_audio, support_labels, query_audio, query_labels = data_item

        if args.mixed_precision:
            with torch.cuda.amp.autocast():
                logits, query_labels = model(
                    support_audio.to(device=device, dtype=torch.float),
                    support_labels.to(device=device, dtype=torch.float),
                    query_audio.to(device=device, dtype=torch.float),
                    query_labels=query_labels.to(device=device, dtype=torch.float)
                )
                loss = loss_fn(logits, query_labels)
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
        else:
            logits, query_labels = model(
                support_audio.to(device=device, dtype=torch.float),
                support_labels.to(device=device, dtype=torch.float),
                query_audio.to(device=device, dtype=torch.float),
                query_labels=query_labels.to(device=device, dtype=torch.float)
            )
            loss = loss_fn(logits, query_labels)
            loss = loss / args.gradient_accumulation_steps

            loss.backward()
        
        acc, prec, rec = compute_metrics(logits, query_labels)
        
        if rank == 0:
            history['loss'].append(loss.item())
            history['learning_rate'].append(optimizer.param_groups[0]["lr"])
            history['accuracy'].append(float(acc.item()))
            history['precision'].append(float(prec.item()))
            history['recall'].append(float(rec.item()))

            if args.wandb:
                wandb.log({
                    "loss": loss.item(),
                    "accuracy": acc.item(),
                    "precision": prec.item(),
                    "recall": rec.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                })

            if t % args.log_steps == 0:
                print(f"Step {t}: Loss={np.mean(history['loss'][-10:])}, Accuracy={np.mean(history['accuracy'][-10:])}, Precision={np.mean(history['precision'][-10:])}, Recall={np.mean(history['recall'][-10:])}")

        if (t + 1) % args.gradient_accumulation_steps == 0 or (t + 1 == len(dataloader)):
            if args.mixed_precision:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scheduler.step()
                optimizer.zero_grad()
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        if rank == 0 and ((t % args.checkpoint_frequency == 0) or (t == len(dataloader) - 1)):
            print(f"Step {t}: Loss={np.mean(history['loss'][-args.checkpoint_frequency:])}, Accuracy={np.mean(history['accuracy'][-args.checkpoint_frequency:])}, Precision={np.mean(history['precision'][-args.checkpoint_frequency:])}, Recall={np.mean(history['recall'][-args.checkpoint_frequency:])}")
            with open(os.path.join(args.experiment_dir, "history.yaml"), 'w') as f:
                yaml.dump(history, f)
            checkpoint_dict = {
                "step": t,
                "model_state_dict": model.state_dict() if single_gpu else model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
            }

            torch.save(checkpoint_dict, os.path.join(args.experiment_dir, f"model_{t}.pt"))
            Path(os.path.join(args.experiment_dir, f"model_{int(t-3*args.checkpoint_frequency)}.pt")).unlink(missing_ok=True)
        
        if (rank == 0) and (t == (len(dataloader) - 1)):
            torch.save(model.state_dict() if single_gpu else model.module.state_dict(), os.path.join(args.experiment_dir, "final_model.pt"))
    
    if not single_gpu:
        cleanup()
    
    return model

if __name__ == "__main__":
    main(sys.argv[1:])
