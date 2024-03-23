import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.hub
import math
from einops import rearrange
from torchaudio.models import wav2vec2_model
import json
from x_transformers import ContinuousTransformerWrapper, Encoder


class AvesEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        config = self.load_config(args.aves_config_fp)
        self.model = wav2vec2_model(**config, aux_num_out=None)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.hub.load_state_dict_from_url(args.aves_url, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.feature_extractor.requires_grad_(False)
        self.sr=args.sr

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)

        return obj

    def forward(self, sig):
        # extract_feature in the torchaudio version will output all 12 layers' output, -1 to select the final one
        out = self.model.extract_features(sig)[0][-1]

        return out
      
    def freeze(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = True

class AvesEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.aves_embedding = AvesEmbedding(args)
        self.args = args
        self.audio_chunk_size_samples = int(args.sr * args.audio_chunk_size_sec)

    def forward(self, x):
        """
        Input
            x (Tensor): (batch, time) (time at 16000 Hz, audio_sr)
        Returns
            x_encoded (Tensor): (batch, embedding_dim, time) (time at 50 Hz, aves_sr)
        """

        # chunk long audio into smaller pieces
        # maybe better to do via a reshape? there is some tradeoff here
        x_encoded = []

        total_len_samples_x = x.size(1)
        x = x-torch.mean(x,axis=1,keepdim=True)

        for start_sample in range(0, total_len_samples_x, self.audio_chunk_size_samples):
            x_sub = x[:,start_sample:start_sample+self.audio_chunk_size_samples]

            if x_sub.size(1) % self.args.scale_factor != 0:
                raise Exception("audio length is not divisible by scale factor")

            expected_dur_output = x_sub.size(1)//self.args.scale_factor

            feats = self.aves_embedding(x_sub)
            feats = rearrange(feats, 'b t c -> b c t')

            #aves may be off by 1 sample from expected
            pad = expected_dur_output - feats.size(2)
            if pad>0:
                feats = F.pad(feats, (0,pad), mode='reflect')

            x_encoded.append(feats)

        x_encoded = torch.cat(x_encoded, dim=2)
        if x_encoded.size(2) != total_len_samples_x / self.args.scale_factor:
            raise Exception("Incorrect feature duration")

        return x_encoded

    def freeze(self):
        self.aves_embedding.freeze()
          
    def unfreeze(self):
        self.aves_embedding.unfreeze()

class LabelEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.support_seq_len = int(args.support_dur_sec * args.sr / args.scale_factor)
        self.query_seq_len = int(args.query_dur_sec * args.sr / args.scale_factor)
        assert args.support_dur_sec * args.sr / args.scale_factor == self.support_seq_len
        assert args.query_dur_sec * args.sr / args.scale_factor == self.query_seq_len

        self.transformer = ContinuousTransformerWrapper(
            dim_in = args.embedding_dim + 1,
            dim_out = 512,
            max_seq_len = int(self.support_seq_len + self.query_seq_len),
            attn_layers = Encoder(
                dim = args.label_encoder_dim,
                depth = args.label_encoder_depth,
                heads = args.label_encoder_heads
            )
        )
        self.args = args

    def forward(self, encoded_support_audio, support_labels_downsampled, encoded_query_audio):
        query_masked_labels = torch.full((encoded_query_audio.size(0), 1, encoded_query_audio.size(2)), -1, device = encoded_query_audio.device)

        support_cat = torch.cat((encoded_support_audio, support_labels_downsampled), dim=1) # (batch, embedding_dim+1, time/scale_factor)
        query_cat = torch.cat((encoded_query_audio, query_masked_labels), dim=1) # (batch, embedding_dim+1, time/scale_factor)

        transformer_input = torch.cat((support_cat, query_cat), dim = 2)
        transformer_input = rearrange(transformer_input, 'b c t -> b t c')

        transformer_output = self.transformer(transformer_input)
        transformer_output = rearrange(transformer_output, 'b t c -> b c t')
        out = transformer_output[:,:,self.support_seq_len:] # return only embedding for query
        return out


class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.audio_encoder = AvesEncoder(args)
        self.label_encoder = LabelEncoder(args)
        self.prediction_head = nn.Conv1d(512, 1, 1)
        self.args = args

    def forward(self, support_audio, support_labels, query_audio, query_labels=None):
        """
        Input
            support_audio (Tensor): (batch, time) (at audio_sr)
            support_labels (Tensor): (batch, time) (at audio_sr)
            query_audio (Tensor): (batch, time) (at audio_sr)
            query_labels (Tensor): (batch, time) (at audio_sr)
        Output
            logits (Tensor): (batch, query_time/scale_factor) (at audio_sr / scale factor)
        """

        encoded_support_audio = self.audio_encoder(support_audio) # (batch, embedding_dim, time) (at audio_sr / scale factor). embedding_dim=768 for aves
        encoded_query_audio = self.audio_encoder(query_audio)

        support_labels = torch.unsqueeze(support_labels, 1) # (batch, 1, time)
        support_labels_downsampled = F.max_pool1d(support_labels, self.args.scale_factor, padding=0) # (batch, 1 , time/scale_factor). 0=NEG 1=UNK 2=POS
        
        if query_labels is not None:
            query_labels = torch.unsqueeze(query_labels, 1) # (batch, 1, time)
            query_labels = F.max_pool1d(query_labels, self.args.scale_factor, padding=0) # (batch, 1 , time/scale_factor). 0=NEG 1=UNK 2=POS
            query_labels = torch.squeeze(query_labels, 1)

        query_representation = self.label_encoder(encoded_support_audio, support_labels_downsampled, encoded_query_audio) # (batch, 512 , query_time/scale_factor). 

        logits = self.prediction_head(query_representation) # (batch, 1, query_time/scale_factor)
        logits = torch.squeeze(logits, dim=1) # (batch, query_time/scale_factor)

        return logits, query_labels

    def freeze_audio_encoder(self):
        self.audio_encoder.freeze()
          
    def unfreeze_audio_encoder(self):
        self.audio_encoder.unfreeze()


    