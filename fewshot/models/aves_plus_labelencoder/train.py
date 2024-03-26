import yaml
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from functools import partial
import os
from einops import rearrange
import yaml
from pathlib import Path
from torchvision.ops import sigmoid_focal_loss
import random

from fewshot.models.aves_plus_labelencoder.model import FewShotModel
from fewshot.models.aves_plus_labelencoder.params import parse_args, save_params
from fewshot.data.data import get_dataloader

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
    
    model=FewShotModel(args)
    
    if args.previous_checkpoint_fp is not None:
        print(f"loading model weights from {args.previous_checkpoint_fp}")
        cp = torch.load(args.previous_checkpoint_fp)
        model.load_state_dict(cp["model_state_dict"])
    
    ## Training
    model = train(model, args) 
    
    print("Training Complete!")

#     Debugging inference
#     from fewshot.models.aves_plus_labelencoder.inference import inference_dcase
#     df = inference_dcase(model, args, '/home/jupyter/fewshot/data/DCASE2022_Development_Set/Development_Set/PB/BUK4_20161011_000804.wav', '/home/jupyter/fewshot/data/DCASE2022_Development_Set/Development_Set/PB/BUK4_20161011_000804.csv')
#     df.to_csv('/home/jupyter/output_test.csv')
    
    ## Evaluation
    # evaluate(trained_model, args)

def get_loss_fn(args):
    def loss_fn(logits, query_labels):
        # query labels: 0: NEG, 1: UNK, 2: POS
        logits=torch.flatten(logits)
        query_labels=torch.flatten(query_labels)
        
        query_binary = torch.minimum(query_labels, torch.ones_like(query_labels)) #2->1
        loss = sigmoid_focal_loss(logits, query_binary) #TODO add in loss hyperparams?
        unknown_mask = query_labels != 1
        loss = loss * unknown_mask # set loss of unknowns to 0
        loss = torch.mean(loss)
        return loss
        
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
  
    loss_fn = get_loss_fn(args)
  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad = True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=0.1, last_epoch=- 1, verbose=False)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs, eta_min=0, last_epoch=- 1, verbose=False)
  
    history={'loss' : [], 'learning_rate' : [], 'accuracy' : [], 'precision':[], 'recall':[]}
    
    dataloader=get_dataloader(args)
    
    model.freeze_audio_encoder()
    
    for t, data_item in enumerate(dataloader):        
        if t == args.unfreeze_encoder_step:
            model.unfreeze_audio_encoder()
        
        support_audio, support_labels, query_audio, query_labels = data_item
        logits, query_labels = model(support_audio.to(device = device, dtype = torch.float), support_labels.to(device = device, dtype = torch.float), query_audio.to(device = device, dtype = torch.float), query_labels=query_labels.to(device = device, dtype = torch.float))
        
        loss = loss_fn(logits, query_labels)
        acc, prec, rec = compute_metrics(logits, query_labels)
        
        history['loss'].append(loss.item())
        history['learning_rate'].append(optimizer.param_groups[0]["lr"])
        history['accuracy'].append(float(acc.item()))
        history['precision'].append(float(prec.item()))
        history['recall'].append(float(rec.item()))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if t % args.checkpoint_frequency == 0:
            print(f"Step {t}: Loss={np.mean(history['loss'][-args.checkpoint_frequency:])}, Accuracy={np.mean(history['accuracy'][-args.checkpoint_frequency:])}, Precision={np.mean(history['precision'][-args.checkpoint_frequency:])}, Recall={np.mean(history['recall'][-args.checkpoint_frequency:])}")
            with open(os.path.join(args.experiment_dir, "history.yaml"), 'w') as f:
                yaml.dump(history, f)
            checkpoint_dict = {
                "step": t,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                }

            torch.save(checkpoint_dict, os.path.join(args.experiment_dir, f"model_{t}.pt"))
            Path(os.path.join(args.experiment_dir, f"model_{int(t-3*args.checkpoint_frequency)}.pt")).unlink(missing_ok=True)

if __name__ == "__main__":
    main(sys.argv[1:])
