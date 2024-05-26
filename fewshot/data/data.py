import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch
import torchaudio

SCENARIO_WEIGHTS = {
    "normal": 1,
    "fine_grained_general": .75,
    "low_snr": 0.75,
    "disjunction_within_species": 0.5,
    "generalization_within_species": 0.5,
    "fine_grained_snr": 0.5,
    "disjunction_cross_species": 0.25,
    "fine_grained_pitch": 0.25,
    "fine_grained_duration": 0.25
}

def load_audio(fp, target_sr):
    audio, file_sr = torchaudio.load(fp)
    
    if file_sr != target_sr:
        print("resampling", fp, file_sr, target_sr)
        audio = torchaudio.functional.resample(audio, file_sr, target_sr)
    
    # correct DC offset
    audio = audio-torch.mean(audio, -1, keepdim=True)
    
    if len(audio.size()) == 2:
        # convert to mono
        audio = torch.mean(audio, dim=0)
    
    return audio

def apply_windowing(audio, labels, chunk_size_samples):
    audio_dur_samples = audio.size(0)
    remainder = audio_dur_samples % chunk_size_samples

    if remainder >= chunk_size_samples // 2:
        to_cut = remainder - chunk_size_samples // 2
    else:
        to_cut = remainder + chunk_size_samples // 2

    halfwindowed_audio_len = audio[to_cut:].size(0)
    assert halfwindowed_audio_len % chunk_size_samples == chunk_size_samples // 2, "Incorrect windowing math!"

    audio = torch.cat((audio[to_cut:], audio))
    labels = torch.cat((labels[to_cut:], labels))

    pad = (chunk_size_samples - (audio.size(0) % chunk_size_samples)) % chunk_size_samples

    if pad > 0:
        audio = torch.cat((audio, audio[:pad]))
        labels = torch.cat((labels, labels[:pad]))

    return audio, labels


class FewshotDataset(Dataset):
    def __init__(self, args):
        self.args = args
        
        # Load audio info csv's
        self.TUT_background_audio_info = pd.read_csv(args.TUT_background_audio_info_fp)
        self.XenoCanto_background_audio_info = pd.read_csv(args.xeno_canto_background_audio_info_fp)
        self.AudioSet_background_audio_info = pd.read_csv(args.audioset_background_audio_info_fp)
        self.pseudovox_info = pd.read_csv(self.args.pseudovox_info_fp)
        self.nonbio_pseudovox_info = pd.read_csv(self.args.nonbio_pseudovox_info_fp)
        
        # Subselect background examples to be used
        self.TUT_background_audio_info = self.TUT_background_audio_info[self.TUT_background_audio_info['duration_sec'] >= self.args.min_background_duration]
        self.XenoCanto_background_audio_info = self.XenoCanto_background_audio_info[self.XenoCanto_background_audio_info['duration_sec'] >= self.args.min_background_duration]
        self.AudioSet_background_audio_info = self.AudioSet_background_audio_info[self.AudioSet_background_audio_info['duration_sec'] >= self.args.min_background_duration]
        
        # Subselect pseudovox to be used
        self.pseudovox_info = self.pseudovox_info[self.pseudovox_info['duration_sec'] <= self.args.max_pseudovox_duration]
        self.pseudovox_info = self.pseudovox_info[self.pseudovox_info['birdnet_confidence'] > self.args.birdnet_confidence_strict_lower_bound]        
        
        fine_clusters_with_enough_examples = pd.Series(sorted(self.pseudovox_info["fp_plus_prediction"].value_counts()[self.pseudovox_info["fp_plus_prediction"].value_counts() >= self.args.min_cluster_size].index)) #fine clusters: birdnet label + origin file
        
        pseudovox_info_long = self.pseudovox_info[(self.pseudovox_info['duration_sec'] >= 1) & (self.pseudovox_info['pseudovox_audio_fp'].str.contains('xeno_canto'))]
        fine_clusters_with_enough_examples_long = pd.Series(sorted(pseudovox_info_long["fp_plus_prediction"].value_counts()[pseudovox_info_long["fp_plus_prediction"].value_counts() >= self.args.min_cluster_size_for_longish_pseudovox].index)) # allow for fewer examples if they are long

        fine_clusters_with_enough_examples = pd.concat([fine_clusters_with_enough_examples, fine_clusters_with_enough_examples_long])
        
        self.pseudovox_info = self.pseudovox_info[self.pseudovox_info['fp_plus_prediction'].isin(fine_clusters_with_enough_examples)]
        
        self.coarse_clusters_with_enough_examples = pd.Series(sorted(self.pseudovox_info[self.pseudovox_info["fp_plus_prediction"].isin(list(fine_clusters_with_enough_examples))]["birdnet_prediction"].unique()))
        
        self.nonbio_pseudovox_info = self.nonbio_pseudovox_info[self.nonbio_pseudovox_info['duration_sec'] <= self.args.max_pseudovox_duration]
        self.nonbio_clusters_with_enough_examples = pd.Series(sorted(self.nonbio_pseudovox_info["fp_plus_prediction"].value_counts()[self.nonbio_pseudovox_info["fp_plus_prediction"].value_counts() >= self.args.nonbio_min_cluster_size].index))
        
        # Init augmentations
        self.shift_up = torchaudio.transforms.PitchShift(16000, 12, hop_length=64)
        self.shift_up2 = torchaudio.transforms.PitchShift(16000, 24, hop_length=64)
        self.shift_down = torchaudio.transforms.PitchShift(16000, -12, hop_length=64)
        self.shift_down2 = torchaudio.transforms.PitchShift(16000, -24, hop_length=64)

        self.scenarios = self.args.scenarios.split(',')
        
        # Init resamplers
        self.resamplers = {}
        self.resamplers[(args.sr, args.sr//2)] = torchaudio.transforms.Resample(orig_freq=args.sr, new_freq=args.sr//2)
        self.resamplers[(args.sr, args.sr*2)] = torchaudio.transforms.Resample(orig_freq=args.sr, new_freq=args.sr*2)

        
        self.audio_chunk_size_samples = int(self.args.audio_chunk_size_sec * self.args.sr)
        
    def load_audio(self, fp, target_sr):
        audio, file_sr = torchaudio.load(fp)

        if file_sr != target_sr:
            if (file_sr, target_sr) in self.resamplers:
                audio = self.resamplers[(file_sr, target_sr)](audio)
            else:
                audio = torchaudio.functional.resample(audio, file_sr, target_sr)

        # correct DC offset
        audio = audio-torch.mean(audio, -1, keepdim=True)

        if len(audio.size()) == 2:
            # convert to mono
            audio = torch.mean(audio, dim=0)

        return audio
        
    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        
        ## choose all randomness
        
        # Special scenario
        
        
        scenario = rng.choice(self.scenarios, p=[SCENARIO_WEIGHTS[s] for s in self.scenarios])
            
        # ["normal", "disjunction_cross_species", "disjunction_within_species", "generalization_within_species", "low_snr", "fine_grained_snr", "fine_grained_pitch", "fine_grained_duration"], p = [0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0, 0])
                
        copy_support = 0 #rng.binomial(1, 0.01)
        
        # Background
            
        if scenario == "low_snr":
            background_source = "TUT"
        else:
            background_source = rng.choice(["TUT", "AudioSet", "XenoCanto"], p = [0.2, 0.2, 0.6])
                        
        if background_source == "TUT":
            background_fps = list(self.TUT_background_audio_info['raw_audio_fp'].sample(n=2, random_state=index))
        elif background_source == "AudioSet":
            background_fps = list(self.AudioSet_background_audio_info['raw_audio_fp'].sample(n=2, random_state=index))
        elif background_source == "XenoCanto":
            background_fps = list(self.XenoCanto_background_audio_info['raw_audio_fp'].sample(n=2, random_state=index))
        
        # Support background
        support_background_fp = background_fps[0]
        support_background_resample = rng.choice(["upsample", "same", "downsample"], p = [0.2, 0.4, 0.4])
        
        # Query background
        background_audio_query_domain_shift = rng.binomial(1, 0.5)
        
        if background_audio_query_domain_shift:
            query_background_resample = rng.choice(["upsample", "same", "downsample"], p = [0.2, 0.4, 0.4])
            query_background_fp = background_fps[1]
        else:
            query_background_resample = support_background_resample
            query_background_fp = background_fps[0]
        
        # prepare which pseudovox to choose from, for adding to background audio
        pseudovox_from_here = self.pseudovox_info[self.pseudovox_info['raw_audio_fp'].isin(background_fps)] #NOTE: filtered based on both backgrounds but sometimes only one is used
        coarse_clusters_present = pseudovox_from_here["birdnet_prediction"].unique()
        coarse_clusters_allowed = self.coarse_clusters_with_enough_examples[~self.coarse_clusters_with_enough_examples.isin(coarse_clusters_present)]
        
        
        if len(coarse_clusters_allowed) < 3:
            # corner case which probably never occurs: one background track contains almost every possible type of sound
            coarse_clusters_allowed = self.coarse_clusters_with_enough_examples
        
        coarse_clusters_to_possibly_include = list(coarse_clusters_allowed.sample(4, random_state=index))
        
        # Focal calls
        focal_rate = rng.choice([5/120, 5/60, 5/30, 5/15])
        focal_pitch = None # don't pitch shift unless in fine-grained scenario
        focal_duration = rng.choice(["long", "same", "short"])
        focal_snr = rng.uniform(-5, 2)
        coarse_focal_c = coarse_clusters_to_possibly_include[2]
        focal_c = list(self.pseudovox_info[self.pseudovox_info["birdnet_prediction"] == coarse_focal_c]["fp_plus_prediction"].sample(1, random_state=index))[0]
        focal_time_reverse = rng.binomial(1, 0.2)
        other_focal_c = focal_c # placeholder
        
        if (scenario == "fine_grained_pitch") or (scenario == "fine_grained_duration"):
            focal_pitch = rng.choice(["up", "same", "down"])
        
        if scenario == "low_snr":
            focal_snr = rng.uniform(-10, -5)
            
        if scenario == "disjunction_cross_species":
            other_coarse_focal_c = coarse_clusters_to_possibly_include[3]
            other_focal_c = list(self.pseudovox_info[self.pseudovox_info["birdnet_prediction"] == other_coarse_focal_c]["fp_plus_prediction"].sample(1, random_state=index))[0]
            focal_rate = focal_rate / 2
            
        if scenario == "disjunction_within_species":
            other_focal_c = list(self.pseudovox_info[self.pseudovox_info["birdnet_prediction"] == coarse_focal_c]["fp_plus_prediction"].sample(1, random_state=index+1))[0]
            focal_rate = focal_rate / 2
            
        if scenario == "generalization_within_species":
            focal_c_df = self.pseudovox_info[self.pseudovox_info["fp_plus_prediction"] == focal_c]
            dur_min_sampled = focal_c_df['duration_sec'].min()
            dur_max_sampled = focal_c_df['duration_sec'].min()
            generalization_df = self.pseudovox_info[(self.pseudovox_info["birdnet_prediction"] == coarse_focal_c) & (self.pseudovox_info["duration_sec"] >= 0.75*dur_min_sampled) & (self.pseudovox_info["duration_sec"] <= 1.25*dur_max_sampled)]
                    
        # Nonfocal calls
        nonfocal_rate = rng.choice([5/120, 5/60, 5/30, 5/15])
        nonfocal_pitch = None
        nonfocal_duration = rng.choice(["long", "same", "short"])
        nonfocal_snr = rng.uniform(-10, 2)
        nonfocal_time_reverse = rng.binomial(1, 0.2)
        
        use_nonbio_as_nonfocal = rng.binomial(1, 0.1)
        
        if use_nonbio_as_nonfocal:
            nonfocal_c = list(self.nonbio_clusters_with_enough_examples.sample(1, random_state=index))[0]
        else:
            coarse_nonfocal_c = coarse_clusters_to_possibly_include[0]
            nonfocal_c = list(self.pseudovox_info[self.pseudovox_info["birdnet_prediction"] == coarse_nonfocal_c]["fp_plus_prediction"].sample(1, random_state=index))[0]
            # nonfocal_c = clusters_to_possibly_include[0]
        
        if scenario == "fine_grained_snr":
            nonfocal_rate = focal_rate
            nonfocal_pitch = focal_pitch
            nonfocal_duration = focal_duration
            nonfocal_snr = focal_snr - rng.uniform(5,10)
            nonfocal_c = focal_c
            use_nonbio_as_nonfocal = False # overwrite
            nonfocal_time_reverse = focal_time_reverse
            
        if scenario == "fine_grained_pitch":
            nonfocal_rate = focal_rate
            nonfocal_pitches = ["up", "same", "down"]
            nonfocal_pitches.remove(focal_pitch)
            nonfocal_pitch = rng.choice(nonfocal_pitches)
            nonfocal_duration = focal_duration
            nonfocal_snr = focal_snr
            nonfocal_c = focal_c
            use_nonbio_as_nonfocal = False  # overwrite
            nonfocal_time_reverse = focal_time_reverse
            
        if scenario == "fine_grained_duration":
            nonfocal_rate = focal_rate
            nonfocal_pitch = focal_pitch
            nonfocal_durations = ["long", "same", "short"]
            nonfocal_durations.remove(focal_duration)
            nonfocal_duration = rng.choice(nonfocal_durations)
            nonfocal_snr = focal_snr
            nonfocal_c = focal_c
            use_nonbio_as_nonfocal = False # overwrite
            nonfocal_time_reverse = focal_time_reverse
            
        if scenario == "fine_grained_general":
            nonfocal_rate = focal_rate
            nonfocal_pitch = focal_pitch
            nonfocal_duration = focal_duration
            nonfocal_snr = focal_snr
            use_nonbio_as_nonfocal = False # overwrite
            nonfocal_time_reverse = focal_time_reverse
            
            # look for sounds of similar duration
            focal_c_df = self.pseudovox_info[self.pseudovox_info["fp_plus_prediction"] == focal_c]
            dur_min_sampled = focal_c_df['duration_sec'].min()
            dur_max_sampled = focal_c_df['duration_sec'].min()
            similar_sounds_df = self.pseudovox_info[(self.pseudovox_info["birdnet_prediction"] != coarse_focal_c) & (self.pseudovox_info["duration_sec"] >= 0.75*dur_min_sampled) & (self.pseudovox_info["duration_sec"] <= 1.25*dur_max_sampled)]
            if len(similar_sounds_df)>0:
                coarse_nonfocal_c = list(similar_sounds_df["birdnet_prediction"].sample(1, random_state=index))[0]
                nonfocal_c = list(self.pseudovox_info[self.pseudovox_info["birdnet_prediction"] == coarse_nonfocal_c]["fp_plus_prediction"].sample(1, random_state=index))[0]
        
        # Load background_audio
        
        r = {"upsample" : int(2*self.args.sr), "same" : self.args.sr, "downsample" : int(0.5*self.args.sr)}
        audio_support = self.load_audio(support_background_fp, r[support_background_resample])
        audio_query = self.load_audio(query_background_fp, r[query_background_resample])
                    
        # loop and trim background audio to desired length
        
        support_dur_samples = int(self.args.support_dur_sec * self.args.sr)
        query_dur_samples = int(self.args.query_dur_sec * self.args.sr)

        # Re-stitch the support background audio to match inference
        restitch_support = rng.binomial(1, 0.1)
        if restitch_support:
            num_stitch_chunks = rng.integers(2, 5)
            gap_size_samples = 100
            min_length = support_dur_samples + num_stitch_chunks * gap_size_samples
            
            if audio_support.size(0) >= min_length:
                chunk_size = (support_dur_samples - gap_size_samples * (num_stitch_chunks - 1)) // num_stitch_chunks
                chunks = [audio_support[i*chunk_size + i*gap_size_samples : (i+1)*chunk_size + i*gap_size_samples] for i in range(num_stitch_chunks)]
                audio_support = torch.cat(chunks, dim=0)
        
        if audio_support.size(0) <= support_dur_samples:
            # corner case: support soundscape is not long enough. in this case, tile it to make it long enough
            audio_support =torch.tile(audio_support, (support_dur_samples//audio_support.size(0)+2,))
            
        if audio_query.size(0) <= query_dur_samples:
            # corner case: query soundscape is not long enough. in this case, tile it to make it long enough
            audio_query =torch.tile(audio_query, (query_dur_samples//audio_query.size(0)+2,))
        
        audio_support_start_sample = rng.integers(0, audio_support.size(0) - support_dur_samples)
        #TODO: query background non-overlapping where possible
        audio_query_start_sample = rng.integers(0, audio_query.size(0) - query_dur_samples)
        
        audio_support = audio_support[audio_support_start_sample:audio_support_start_sample+support_dur_samples]
        audio_query = audio_query[audio_query_start_sample:audio_query_start_sample+query_dur_samples]
        
        # set up labels
        support_labels = torch.zeros_like(audio_support)
        query_labels = torch.zeros_like(audio_query)
                
        # normalize rms of support and query background audio
        rms_background_audio_support = torch.std(audio_support)
        if torch.std(audio_query) > 0:
            audio_query = audio_query * rms_background_audio_support / torch.std(audio_query)
        
        if (scenario == "disjunction_cross_species") or (scenario == "disjunction_within_species"):
            s_to_iter_through = [2.1,2,0]
        else:
            s_to_iter_through = [2,0]
        
        for i, s in enumerate(s_to_iter_through):
            # s semantics:
            # 2.1 = Other Positive (disjunction scenario)
            # 2 = Positive
            # 1 = Unknown
            # 0 = Negative; note there are already implicit negatives in background track
            
            label = {2.1:2, 2:2, 1:1, 0:0}[s]
            
            # get number of pseudovox to use
            
            pseudovox_rate = {2: focal_rate, 0: nonfocal_rate}[label] # call rate per second
            
            n_pseudovox_support = rng.poisson(pseudovox_rate*self.args.support_dur_sec)
            n_pseudovox_query = rng.poisson(pseudovox_rate*self.args.query_dur_sec)
            
            if label == 2:
                # Require minimum 1 focal call in support
                n_pseudovox_support = max(n_pseudovox_support, 1)
                # if (scenario == "fine_grained_snr") or (scenario == "fine_grained_pitch") or (scenario == "fine_grained_duration"):
                #     n_pseudovox_support = max(n_pseudovox_support, 1)
                    
                # Increase rate in query to reduce number of empty examples
                floor_query = rng.choice([0,1],p=[.75, .25])
                n_pseudovox_query = max(n_pseudovox_query, floor_query)
                
            if label == 0:
                if (scenario == "fine_grained_snr") or (scenario == "fine_grained_pitch") or (scenario == "fine_grained_duration") or (scenario == "fine_grained_general"):
                    n_pseudovox_support = max(n_pseudovox_support, 1)
            
            # get the exact pseudovox to insert
            c = {2.1: other_focal_c, 2: focal_c, 0:nonfocal_c}[s]
            
            if use_nonbio_as_nonfocal and (label==0):
                possible_pseudovox = self.nonbio_pseudovox_info[self.nonbio_pseudovox_info["fp_plus_prediction"] == c]
            else:
                possible_pseudovox = self.pseudovox_info[self.pseudovox_info["fp_plus_prediction"] == c]
            
            
            if (label == 2) & (scenario == "generalization_within_species"):
                pseudovox_query = generalization_df.sample(n=n_pseudovox_query, replace=True, random_state=index+1)
            else:
                pseudovox_query = possible_pseudovox.sample(n=n_pseudovox_query, replace=True, random_state=index+1)
            
            pseudovox_fps_in_query = list(pseudovox_query['pseudovox_audio_fp'].unique()) # don't re-use pseudovox from query in support
            possible_pseudovox_not_in_query = possible_pseudovox[~possible_pseudovox['pseudovox_audio_fp'].isin(pseudovox_fps_in_query)]
            
            if len(possible_pseudovox_not_in_query) >= 2:
                pseudovox_support = possible_pseudovox_not_in_query.sample(n=n_pseudovox_support, replace=True, random_state=index)
            else:
                pseudovox_support = possible_pseudovox.sample(n=n_pseudovox_support, replace=True, random_state=index)
            
            # load the pseudovox and insert them
            for _, row in pseudovox_support.iterrows():
                
                dur_aug = {2: focal_duration, 0: nonfocal_duration}[label]
                pitch_aug = {2: focal_pitch, 0: nonfocal_pitch}[label]
                time_reverse_aug = {2:focal_time_reverse, 0:nonfocal_time_reverse}[label]
                
                speed_adjust_rate = {"long" : int(2*self.args.sr), "same" : self.args.sr, "short" : int(0.5*self.args.sr)}[dur_aug]
                pseudovox = self.load_audio(row['pseudovox_audio_fp'], speed_adjust_rate)
                
                if pitch_aug is not None:
                    with torch.no_grad():
                        if dur_aug == "long":
                            if pitch_aug == "same":
                                pseudovox = self.shift_up(pseudovox)
                            elif pitch_aug == "up":
                                pseudovox = self.shift_up2(pseudovox)

                        if dur_aug == "same":
                            if pitch_aug == "down":
                                pseudovox = self.shift_down(pseudovox)
                            elif pitch_aug == "up":
                                pseudovox = self.shift_up(pseudovox)

                        if dur_aug == "short":
                            if pitch_aug == "same":
                                pseudovox = self.shift_down(pseudovox)
                            elif pitch_aug == "down":
                                pseudovox = self.shift_down2(pseudovox)
                
                rms_pseudovox = torch.std(pseudovox)
                snr_db = {2: focal_snr, 0: nonfocal_snr}[label] + rng.uniform(-1, 1)
                pseudovox = pseudovox * (rms_background_audio_support / rms_pseudovox) * (10**(.1 * snr_db))
                
                if time_reverse_aug:
                    pseudovox = torch.flip(pseudovox, (0,))

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

            for _, row in pseudovox_query.iterrows():
                dur_aug = {2: focal_duration, 0: nonfocal_duration}[label]
                pitch_aug = {2: focal_pitch, 0: nonfocal_pitch}[label]
                
                speed_adjust_rate = {"long" : int(2*self.args.sr), "same" : self.args.sr, "short": int(0.5*self.args.sr)}[dur_aug]
                pseudovox = self.load_audio(row['pseudovox_audio_fp'], speed_adjust_rate)
                
                if pitch_aug is not None:
                    with torch.no_grad():
                        if dur_aug == "long":
                            if pitch_aug == "same":
                                pseudovox = self.shift_up(pseudovox)
                            elif pitch_aug == "up":
                                pseudovox = self.shift_up2(pseudovox)

                        if dur_aug == "same":
                            if pitch_aug == "down":
                                pseudovox = self.shift_down(pseudovox)
                            elif pitch_aug == "up":
                                pseudovox = self.shift_up(pseudovox)

                        if dur_aug == "short":
                            if pitch_aug == "same":
                                pseudovox = self.shift_down(pseudovox)
                            elif pitch_aug == "down":
                                pseudovox = self.shift_down2(pseudovox)
                
                rms_pseudovox = torch.std(pseudovox)
                snr_db = {2: focal_snr, 0: nonfocal_snr}[label] + rng.uniform(-1, 1)
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
                
        if copy_support:
            slice_start_sample = rng.integers(0, support_dur_samples-query_dur_samples)
            audio_query = audio_support[slice_start_sample:slice_start_sample+query_dur_samples]
            query_labels = support_labels[slice_start_sample:slice_start_sample+query_dur_samples]

        if self.args.window_train_support:
            audio_support, support_labels = apply_windowing(audio_support, support_labels, self.audio_chunk_size_samples)
        
        
        
        


        return audio_support, support_labels, audio_query, query_labels


    def __len__(self):
        return self.args.n_synthetic_examples
      
def get_dataloader_distributed(dataset, args, world_size, rank):
    train_dataloader = DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last = False,
                                  sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank))
    return train_dataloader

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
        hop_ratio = 0.5
        self.args = args
        self.support_audio = support_audio
        self.support_labels = support_labels
        self.query_audio = query_audio
        self.query_dur_samples = int(args.sr * args.query_dur_sec)
        assert self.query_dur_samples*hop_ratio == int(self.query_dur_samples*hop_ratio) # hop ratio must divide window size
        self.hop_samples = int(self.query_dur_samples*hop_ratio)
        assert self.query_audio.size(0) >= self.query_dur_samples # audio must be at least one window long
        assert (self.query_audio.size(0) - self.query_dur_samples) % self.hop_samples == 0 # hop size must divide audio length
        
    def __getitem__(self, index):
        return self.support_audio, self.support_labels, self.query_audio[index * self.hop_samples : index * self.hop_samples + self.query_dur_samples]
    
    def __len__(self):
        return 1 + (self.query_audio.size(0) - self.query_dur_samples) // self.hop_samples
    
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
    import tarfile
    
    args = sys.argv[1:]
    
    # set output dir
    
    output_dir = 'fewshot_demo_clips_rss2'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # parse args

    parser = argparse.ArgumentParser()

    # General    
    
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--audio-chunk-size-sec', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--n-synthetic-examples', type=int, default=20, help="limit on number of unique examples the dataloader will generate; required by pytorch Dataloder")
    parser.add_argument('--support-dur-sec', type=float, default=24, help="dur of support audio fed into model")
    parser.add_argument('--query-dur-sec', type=float, default=4, help="dur of query audio fed into model")
    parser.add_argument('--min-background-duration', type=float, default = 6, help = "the min dur in seconds that a file is allowed to be, in order for it to be used as background audio.")
    parser.add_argument('--pseudovox-info-fp', type=str, default='/home/jupyter/data/fewshot_data/data_medium/pseudovox_bio.csv')
    parser.add_argument('--nonbio-pseudovox-info-fp', type=str, default='/home/jupyter/data/fewshot_data/data_medium/pseudovox_nonbio.csv')
    parser.add_argument('--max-pseudovox-duration', type=float, default=12, help= "the max dur in seconds that a pseudovox may be")
    parser.add_argument('--min-cluster-size', type = int, default=4, help="the minimum number of pseudovox in a cluster, in order for that cluster to be included as an option")
    parser.add_argument('--min-cluster-size-for-longish-pseudovox', type = int, default=2, help = "the min cluster size when a pseudovox is >=1 sec long, we allow this because there aren't that many of them")
    parser.add_argument('--nonbio-min-cluster-size', type = int, default=4, help="the minimum number of nonbio pseudovox in a cluster, in order for that cluster to be included as an option")
    parser.add_argument('--birdnet-confidence-strict-lower-bound', type=float, default=0, help="will filter out examples with birdnet confidence <= this value. Mostly used to remove pseudovox with no sounds of interest")
    parser.add_argument('--scenarios', type=str, default="normal,disjunction_cross_species,disjunction_within_species,generalization_within_species,low_snr,fine_grained_snr,fine_grained_pitch,fine_grained_duration,fine_grained_general", help = "csv of scenarios to choose from for constructing examples")
    parser.add_argument('--window-train-support', action='store_true', help="whether to apply windowing to support audio during training")
    
    parser.add_argument('--TUT-background-audio-info-fp', type = str, default='/home/jupyter/data/fewshot_data/data_medium/TUT_background_audio_info.csv')
    parser.add_argument('--audioset-background-audio-info-fp', type = str, default='/home/jupyter/data/fewshot_data/data_medium/audioset_background_audio_info.csv')
    parser.add_argument('--xeno-canto-background-audio-info-fp', type = str, default='/home/jupyter/data/fewshot_data/data_medium/xeno_canto_background_audio_info.csv')
    
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
            else:
                end = len(labels)-1
            d["Begin Time (s)"].append(start/args.sr)
            d["End Time (s)"].append(end/args.sr)
            d["Annotation"].append("POS")
                
        if labels[0]:
            start = 0
            look_forward = labels[start:]
            ends = np.where(~look_forward)[0]
            if len(ends)>0:
                end = start+np.amin(ends)
            else:
                end = len(labels)-1
            d["Begin Time (s)"].append(start/args.sr)
            d["End Time (s)"].append(end/args.sr)
            d["Annotation"].append("POS")

        d = pd.DataFrame(d)
        d.to_csv(os.path.join(output_dir, f"selection_table_{i}.txt"), sep='\t', index=False)
    
    # Tar the output directory
    tar_path = f"{output_dir}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(output_dir, arcname=os.path.basename(output_dir))
    
    print(f"Output directory {output_dir} has been archived to {tar_path}")
