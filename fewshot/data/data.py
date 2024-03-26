import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

import torch
import torchaudio


def load_audio(fp, target_sr):
    audio, file_sr = torchaudio.load(fp)
    
    if file_sr != target_sr:
        audio = torchaudio.functional.resample(audio, file_sr, target_sr)
    
    # correct DC offset
    audio = audio-torch.mean(audio, -1, keepdim=True)
    
    if len(audio.size()) == 2:
        # convert to mono
        audio = torch.mean(audio, dim=0)
    
    return audio

class FewshotDataset(Dataset):
    def __init__(self, args):
        self.args = args
        
        # Load audio info csv's
        self.background_audio_info = pd.read_csv(args.background_audio_info_fp)
        self.pseudovox_info = pd.read_csv(self.args.pseudovox_info_fp)
        
        # Subselect background examples to be used
        self.background_audio_info = self.background_audio_info[self.background_audio_info['duration_sec'] >= self.args.min_background_duration]
        
        # Subselect pseudovox to be used
        self.pseudovox_info = self.pseudovox_info[self.pseudovox_info['duration_sec'] <= self.args.max_pseudovox_duration]
        self.pseudovox_info = self.pseudovox_info[self.pseudovox_info['birdnet_confidence'] > self.args.birdnet_confidence_strict_lower_bound]
        self.clusters_with_enough_examples = pd.Series(sorted(self.pseudovox_info[self.args.cluster_column].value_counts()[self.pseudovox_info[self.args.cluster_column].value_counts() >= self.args.min_cluster_size].index))
    
    def get_pseudovox_rate(self, label, rng):
        # return rate in pseudovox / second
        # hard-coded for now
        
        if label == 0:
            return 5/30
        
        if label == 1:
            return 0
        
        if label == 2:
            return 5/30
        
    def get_snr_db(self, label, rng):
        # return snr in dB
        # hard coded for now
        
        return rng.uniform(0, 2)
        
    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        
        # choose background audio
        
        background_audio_fps = list(self.background_audio_info['raw_audio_fp'].sample(n=2, random_state=index))
        background_audio_fps = [os.path.join(self.args.background_audio_dir, os.path.basename(x)) for x in background_audio_fps]
        audio_support = load_audio(background_audio_fps[0], self.args.sr)
        audio_query = load_audio(background_audio_fps[1], self.args.sr)
        
        # loop and trim background audio to desired length
        
        support_dur_samples = int(self.args.support_dur_sec * self.args.sr)
        query_dur_samples = int(self.args.query_dur_sec * self.args.sr)
        
        if audio_support.size(0) <= support_dur_samples:
            # corner case: support soundscape is not long enough. in this case, tile it to make it long enough
            audio_support =torch.tile(audio_support, (support_dur_samples//audio_support.size(0)+2,))
            
        if audio_query.size(0) < query_dur_samples:
            # corner case: query soundscape is not long enough. in this case, tile it to make it long enough
            audio_query =torch.tile(audio_query, (query_dur_samples//audio_query.size(0)+2,))
        
        audio_support_start_sample = rng.integers(0, audio_support.size(0) - support_dur_samples)
        audio_query_start_sample = rng.integers(0, audio_query.size(0) - query_dur_samples)
        
        audio_support = audio_support[audio_support_start_sample:audio_support_start_sample+support_dur_samples]
        audio_query = audio_query[audio_query_start_sample:audio_query_start_sample+query_dur_samples]
        
        # set up labels and mask
        
        support_labels = torch.zeros_like(audio_support)
        query_labels = torch.zeros_like(audio_query)
        
        support_label_mask = torch.ones_like(audio_support) # model DOES see labels from support
        query_label_mask = torch.zeros_like(audio_query) # model DOES NOT see labels from query
        
        # normalize rms of support and query background audio
        
        rms_background_audio_support = torch.std(audio_support)
        if torch.std(audio_query) > 0:
            audio_query = audio_query * rms_background_audio_support / torch.std(audio_query)
                             
        # prepare which pseudovox to choose from, for adding to background audio
        
        pseudovox_from_here = self.pseudovox_info[self.pseudovox_info['raw_audio_fp'].isin(background_audio_fps)]
        clusters_present = pseudovox_from_here[self.args.cluster_column].unique()
        clusters_allowed = self.clusters_with_enough_examples[~self.clusters_with_enough_examples.isin(clusters_present)]
        
        if len(clusters_allowed) < 3:
            # corner case which probably never occurs: one background track contains almost every possible type of sound
            clusters_allowed = self.clusters_with_enough_examples
        
        clusters_to_possibly_include = list(clusters_allowed.sample(3, random_state=index))
        
        # add pseudovox into clips
        
        for i, label in enumerate([2,1,0]):
            # label semantics:
            # 2 = Positive
            # 1 = Unknown
            # 0 = Negative; note there are already implicit negatives in background track
            
            # get number of pseudovox to use
            
            pseudovox_rate = self.get_pseudovox_rate(label, rng) # call rate per second
            
            n_pseudovox_support = rng.poisson(pseudovox_rate*self.args.support_dur_sec)
            n_pseudovox_query = rng.poisson(pseudovox_rate*self.args.query_dur_sec)
            
            if label == 2:
                # Require minimum 1 focal call in support
                n_pseudovox_support = max(n_pseudovox_support, 1)
            
            # get the exact pseudovox to insert
            
            c = clusters_to_possibly_include[i]
            possible_pseudovox = self.pseudovox_info[self.pseudovox_info[self.args.cluster_column] == c]

            pseudovox_support = possible_pseudovox.sample(n=n_pseudovox_support, replace=True, random_state=index)
            pseudovox_query = possible_pseudovox.sample(n=n_pseudovox_query, replace=True, random_state=index+1)
            
            # Choose parameters for processing pseudovox by sampling from a distribution
            
            snr_db = self.get_snr_db(label, rng) # sample snr in dB
            # TODO add others, e.g. pitch shift
            
            # load the pseudovox and insert them
            
            for _, row in pseudovox_support.iterrows():
                pseudovox = load_audio(os.path.join(self.args.pseudovox_audio_dir, os.path.basename(row['filepath'])), self.args.sr)
                
                rms_pseudovox = torch.std(pseudovox)
                
                ##
                # TODO: modify snr_db by some amount so it is not the same for each pseudovox in c
                ##
                
                pseudovox = pseudovox * (rms_background_audio_support / rms_pseudovox) * (10**(.1 * snr_db))
                
                ##
                # TODO: augmentations for the pseudovox, e.g. pitch shift
                ##

                pseudovox_start = rng.integers(-pseudovox.size(0), support_dur_samples)
                
                if pseudovox_start < 0:
                    # corner case: pseudovox is cut off by beginning of clip
                    pseudovox = pseudovox[-pseudovox_start:]
                    pseudovox_start = 0
                    
                if pseudovox_start >= support_dur_samples - pseudovox.size(0):
                    # corner case: pseudovox is cut off by end of clip
                    pseudovox = pseudovox[:support_dur_samples - pseudovox_start]
                
                audio_support[pseudovox_start:pseudovox_start+pseudovox.size(0)] += pseudovox
                support_labels[pseudovox_start:pseudovox_start+pseudovox.size(0)] = torch.maximum(support_labels[pseudovox_start:pseudovox_start+pseudovox.size(0)], torch.full_like(support_labels[pseudovox_start:pseudovox_start+pseudovox.size(0)], label))
                
            for _, row in pseudovox_query.iterrows():
                pseudovox = load_audio(os.path.join(self.args.pseudovox_audio_dir, os.path.basename(row['filepath'])), self.args.sr)
                
                rms_pseudovox = torch.std(pseudovox)
                
                ##
                # TODO: modify snr_db by some amount so it is not the same for each pseudovox in c
                ##
                
                pseudovox = pseudovox * (rms_background_audio_support / rms_pseudovox) * (10**(.1 * snr_db))
                                
                ##
                # TODO: augmentations for the pseudovox, e.g. pitch shift
                ##

                pseudovox_start = rng.integers(-pseudovox.size(0), query_dur_samples)
                
                if pseudovox_start < 0:
                    # corner case: pseudovox is cut off by beginning of clip
                    pseudovox = pseudovox[-pseudovox_start:]
                    pseudovox_start = 0
                    
                if pseudovox_start >= query_dur_samples - pseudovox.size(0):
                    # corner case: pseudovox is cut off by end of clip
                    pseudovox = pseudovox[:query_dur_samples - pseudovox_start]
                
                audio_query[pseudovox_start:pseudovox_start+pseudovox.size(0)] += pseudovox
                query_labels[pseudovox_start:pseudovox_start+pseudovox.size(0)] = torch.maximum(query_labels[pseudovox_start:pseudovox_start+pseudovox.size(0)], torch.full_like(query_labels[pseudovox_start:pseudovox_start+pseudovox.size(0)], label))

        return audio_support, support_labels, audio_query, query_labels

# OLD, DEMO USAGE
#         # glue support and query together

#         audio = torch.cat([audio_support, audio_query])
#         labels = torch.cat([support_labels, query_labels])
#         label_mask = torch.cat([support_label_mask, query_label_mask])

#         return audio, labels, label_mask        

    def __len__(self):
        return self.args.n_synthetic_examples
      
def get_dataloader(args, shuffle = True):
    dataset = FewshotDataset(args)

    train_dataloader = DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=shuffle,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last = False)

    return train_dataloader


class InferenceDataset(Dataset):
    def __init__(self, support_audio, support_labels, query_audio, args):
        # dataloader for audio during few shot inference; 
        # chunks query audio into pieces the be fed into model
        # assumes audio and labels has already been padded if necessary
        # support_audio (Tensor) : (support_dur_samples,)
        # support_labels (Tensor) : (support_dur_samples,)
        # query_audio (Tensor) : (query_dur_samples)
        self.args = args
        self.support_audio = support_audio
        self.support_labels = support_labels
        self.query_audio = query_audio
        self.query_dur_samples = int(args.sr * args.query_dur_sec)
        
    def __getitem__(self, index):
        return self.support_audio, self.support_labels, self.query_audio[index * self.query_dur_samples : (index+1) * self.query_dur_samples]
    
    def __len__(self):
        return int(np.ceil(self.query_audio.size(0) / self.query_dur_samples))
    
def get_inference_dataloader(support_audio, support_labels, query_audio, args):
    dataset = InferenceDataset(support_audio, support_labels, query_audio, args)
    
    inference_dataloader = DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=False,
                                     )
    return inference_dataloader


if __name__ == "__main__":
    # demo usage
    
    import argparse
    import sys
    
    args = sys.argv[1:]
    
    # set output dir
    
    output_dir = '/home/jupyter/fewshot_demo_clips'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # parse args

    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--background-audio-dir', type = str, default = '/home/jupyter/fewshot/data/data_small/audio_trimmed', help="path to dir with background audio")
    parser.add_argument('--pseudovox-audio-dir', type = str, default = '/home/jupyter/fewshot/data/data_small/pseudovox', help="path to dir with pseudovox audio")
    
    parser.add_argument('--background-audio-info-fp', type = str, default='/home/jupyter/fewshot/data/data_small/background_audio_info.csv')
    parser.add_argument('--pseudovox-info-fp', type=str, default='/home/jupyter/fewshot/data/data_small/pseudovox_manifest_birdnet_filtered.csv')
    
    parser.add_argument('--min-background-duration', type=float, default = 15, help = "the min dur in seconds that a file is allowed to be, in order for it to be used as background audio.")
    parser.add_argument('--max-pseudovox-duration', type=float, default=5, help= "the max dur in seconds that a pseudovox may be")
    parser.add_argument('--min-cluster-size', type = int, default=10, help="the minimum number of pseudovox in a cluster, in order for that cluster to be included as an option")
    parser.add_argument('--birdnet-confidence-strict-lower-bound', type=float, default=0, help="will filter out examples with birdnet confidence <= this value. Mostly used to remove pseudovox with no sounds of interest")
    parser.add_argument('--cluster-column', type=str, default='prediction', choices=['prediction', 'birdnet_prediction'], help="name of column in manifest to use for forming groups of calls")
    
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--n-synthetic-examples', type=int, default=10, help="limit on number of unique examples the dataloader will generate; required by pytorch Dataloder")
    parser.add_argument('--support-dur-sec', type=float, default=30, help="dur of support audio fed into model")
    parser.add_argument('--query-dur-sec', type=float, default=6, help="dur of query audio fed into model")
    
    
    args = parser.parse_args(args)
    
    
    dataloader = get_dataloader(args, shuffle = False)
    
    for i, (support_audio, support_labels, query_audio, query_labels) in enumerate(dataloader):
        torchaudio.save(os.path.join(output_dir, f"audio_{i}.wav"), torch.cat([support_audio, query_audio], dim=1), args.sr)
        np.save(os.path.join(output_dir, f"labels_{i}.npy"), torch.cat([support_labels, query_labels], dim=1).numpy())        
    