import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

import torch
import torchaudio

birdnet_nonbiological_labels = ["Engine", "Fireworks", "Gun", "Noise", "Siren"]

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
        self.pseudovox_info = self.pseudovox_info[~self.pseudovox_info['birdnet_prediction'].isin(birdnet_nonbiological_labels)]
        self.clusters_with_enough_examples = pd.Series(sorted(self.pseudovox_info[self.args.cluster_column].value_counts()[self.pseudovox_info[self.args.cluster_column].value_counts() >= self.args.min_cluster_size].index))
        
        # Init augmentations
        resample_rates = self.args.resample_rates.split(',')
        self.resample_rates = [int(self.args.sr*float(x)) for x in resample_rates]
    
    def get_pseudovox_rate(self, label, rng):
        # return rate in pseudovox / second
        # hard-coded for now
        
        if label == 0:
            return rng.choice([5/60, 5/30, 5/15])
        
        if label == 1:
            return 0
        
        if label == 2:
            return rng.choice([5/60, 5/30, 5/15])
        
    def get_snr_db(self, label, rng):
        # return snr in dB
        return rng.uniform(self.args.snr_db_low, self.args.snr_db_high)
        
    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        # choose background audio
        
        background_audio_fps = list(self.background_audio_info['raw_audio_fp'].sample(n=2, random_state=index))
        background_audio_fps = [os.path.join(self.args.background_audio_dir, os.path.basename(x)) for x in background_audio_fps]
        speed_adjust_rate = rng.choice(self.resample_rates)
        audio_support = load_audio(background_audio_fps[0], speed_adjust_rate)
        
        background_audio_query_domain_shift = rng.binomial(1, self.args.p_background_audio_query_domain_shift)
        
        if background_audio_query_domain_shift==1:
            audio_query = load_audio(background_audio_fps[1], speed_adjust_rate)
        else:
            # if no domain shift, re-use background audio
            audio_query = load_audio(background_audio_fps[0], speed_adjust_rate)

                    
        # loop and trim background audio to desired length
        
        support_dur_samples = int(self.args.support_dur_sec * self.args.sr)
        query_dur_samples = int(self.args.query_dur_sec * self.args.sr)
        
        if audio_support.size(0) <= support_dur_samples:
            # corner case: support soundscape is not long enough. in this case, tile it to make it long enough
            audio_support =torch.tile(audio_support, (support_dur_samples//audio_support.size(0)+2,))
            
        if audio_query.size(0) <= query_dur_samples:
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

        
        # special scenarios:
        
        scenario = rng.choice(['normal', 'fine_grained_amplitude'], p=[0.95, 0.05])
        
        # add pseudovox into clips
        pseudovox_timestamps_support = []
        pseudovox_timestamps_query = []
        
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
            speed_adjust_rate = rng.choice(self.resample_rates)
            
            if scenario=="fine_grained_amplitude" and label==0:
                c=c_focal
                snr_db=snr_db_focal-rng.uniform(2,4)
                speed_adjust_rate = speed_adjust_rate_focal
            
            # Store info for focal, for special scenarios
            
            if label==2:
                c_focal=c
                snr_db_focal=snr_db
                speed_adjust_rate_focal = speed_adjust_rate
            
            # load the pseudovox and insert them
                        
            for _, row in pseudovox_support.iterrows():
                pseudovox = load_audio(os.path.join(self.args.pseudovox_audio_dir, os.path.basename(row['filepath'])), speed_adjust_rate)
                
                rms_pseudovox = torch.std(pseudovox)                
                pseudovox = pseudovox * (rms_background_audio_support / rms_pseudovox) * (10**(.1 * snr_db))

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

                pseudovox_end = min(pseudovox_start + pseudovox.size(0), support_dur_samples)

                # Record the timestamp (convert from samples to seconds)

                pseudovox_timestamps_support.append((pseudovox_start / self.args.sr, pseudovox_end / self.args.sr))
                
            for _, row in pseudovox_query.iterrows():
                pseudovox = load_audio(os.path.join(self.args.pseudovox_audio_dir, os.path.basename(row['filepath'])), speed_adjust_rate)
                
                rms_pseudovox = torch.std(pseudovox)
                
                
                pseudovox = pseudovox * (rms_background_audio_support / rms_pseudovox) * (10**(.1 * snr_db))

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

        pseudovox_timestamps_support = sorted(pseudovox_timestamps_support, key=lambda x: x[0])  # Sort by start time
        pseudovox_timestamps_query = sorted(pseudovox_timestamps_query, key=lambda x: x[0])  # Sort by start time

        if self.args.return_timestamps:
            return audio_support, support_labels, audio_query, query_labels, pseudovox_timestamps_support, pseudovox_timestamps_query

        return audio_support, support_labels, audio_query, query_labels


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
    parser.add_argument('--cluster-column', type=str, default='birdnet_prediction', choices=['prediction', 'birdnet_prediction'], help="name of column in manifest to use for forming groups of calls")
    
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--n-synthetic-examples', type=int, default=25, help="limit on number of unique examples the dataloader will generate; required by pytorch Dataloder")
    parser.add_argument('--support-dur-sec', type=float, default=30, help="dur of support audio fed into model")
    parser.add_argument('--query-dur-sec', type=float, default=6, help="dur of query audio fed into model")
    parser.add_argument('--return-timestamps', type=bool, default=False, help="True to return timestamps from the dataloader")
    
    # Augmentations
    parser.add_argument('--p-background-audio-query-domain-shift', default=0.5, type=float, help="probability of using a different clip for query background audio")
    parser.add_argument('--snr-db-low', default=-4, type=float, help="low value of snr dB")
    parser.add_argument('--snr-db-high', default=2, type=float, help="high value of snr dB")
    parser.add_argument('--resample-rates', default="0.5,1.0,2.0", type=str, help="csv of factors to resample audio at, for augmentations")
    
    args = parser.parse_args(args)
    
    
    dataloader = get_dataloader(args, shuffle = False)
    
    for i, (support_audio, support_labels, query_audio, query_labels) in enumerate(dataloader):
        torchaudio.save(os.path.join(output_dir, f"audio_{i}.wav"), torch.cat([support_audio, query_audio], dim=1), args.sr)
        np.save(os.path.join(output_dir, f"labels_{i}.npy"), torch.cat([support_labels, query_labels], dim=1).numpy())     
        
        #make selection table
        labels=(torch.cat([support_labels, query_labels], dim=1).squeeze(0).numpy()==2)
        starts = labels[1:] * ~labels[:-1]
        starts = np.where(starts)[0] + 1

        d = {"Begin Time (s)" : [], "End Time (s)" : [], "Annotation" : []}

        for start in starts:
            look_forward = labels[start:]
            ends = np.where(~look_forward)[0]
            if len(ends)>0:
                end = start+np.amin(ends)
                d["Begin Time (s)"].append(start/args.sr)
                d["End Time (s)"].append(end/args.sr)
                d["Annotation"].append("POS")

        d = pd.DataFrame(d)
        d.to_csv(os.path.join(output_dir, f"selection_table_{i}.txt"), sep='\t', index=False)
    