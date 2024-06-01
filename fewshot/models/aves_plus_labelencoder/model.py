import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.hub
import math
from einops import rearrange
import torchaudio
from torchaudio.models import wav2vec2_model
import json
from x_transformers import ContinuousTransformerWrapper, Encoder

from fewshot.models.audio_llms.htsat.model import HTSATConfig, create_htsat_model
from fewshot.models.aves_plus_labelencoder.frame_atst import get_timestamp_embedding, load_model

ADD_LABEL_EMBEDDING = True

class AvesEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        # config = self.load_config(args.aves_config_fp)
        # self.model = wav2vec2_model(**config, aux_num_out=None)
        # state_dict = torch.hub.load_state_dict_from_url(args.aves_url, map_location=device)
        # self.model.load_state_dict(state_dict)
        # self.model.feature_extractor.requires_grad_(False)
        
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.model = bundle.get_model()
        
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
        self.model.feature_extractor.requires_grad_(False)

    def unfreeze(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        self.model.feature_extractor.requires_grad_(True)
        
class LabelEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.support_seq_len = int(args.support_dur_sec * args.sr / args.scale_factor)
        self.query_seq_len = int(args.query_dur_sec * args.sr / args.scale_factor)
        assert args.support_dur_sec * args.sr / args.scale_factor == self.support_seq_len
        assert args.query_dur_sec * args.sr / args.scale_factor == self.query_seq_len
        self.transformer = ContinuousTransformerWrapper(
            dim_in = args.embedding_dim if ADD_LABEL_EMBEDDING else args.embedding_dim+4,
            dim_out = 512,
            max_seq_len = int(self.support_seq_len + self.query_seq_len),
            attn_layers = Encoder(
                dim = args.label_encoder_dim,
                depth = args.label_encoder_depth,
                heads = args.label_encoder_heads,
                attn_flash=True,
                rotary_pos_emb = True,
                ff_swish = True,
                ff_glu = True,
            )
        )
        if ADD_LABEL_EMBEDDING:
            self.label_embedding = torch.nn.Linear(4, args.embedding_dim)
        else:
            self.label_embedding=torch.nn.Conv1d(4, 4, 1)
        self.logits_head = nn.Conv1d(512, 1, 1)
        # self.confidences_head = nn.Conv1d(512, 64, 1)
        self.args = args

    def forward(self, encoded_support_audio, support_labels_downsampled, encoded_query_audio):
        support_len = encoded_support_audio.size(2)
        
        support_labels_downsampled=F.one_hot(support_labels_downsampled.long(), num_classes=4).float()
        support_labels_downsampled=rearrange(support_labels_downsampled, 'b t c -> b c t')
        
        query_masked_labels = torch.full((encoded_query_audio.size(0), encoded_query_audio.size(2)), 3, device = encoded_query_audio.device, dtype=torch.long) # 3=masked
        query_masked_labels = F.one_hot(query_masked_labels, num_classes=4).float()
        query_masked_labels = rearrange(query_masked_labels, 'b t c -> b c t')
        

        if ADD_LABEL_EMBEDDING:
            support_labels_downsampled=self.label_embedding(support_labels_downsampled.reshape(-1,4)).reshape(encoded_support_audio.size(0), self.args.embedding_dim, -1)
            query_masked_labels=self.label_embedding(query_masked_labels.reshape(-1,4)).reshape(encoded_query_audio.size(0), self.args.embedding_dim, -1)
        else:
            support_labels_downsampled = self.label_embedding(support_labels_downsampled)
            query_masked_labels = self.label_embedding(query_masked_labels)
        
        
        if ADD_LABEL_EMBEDDING:
            support_cat = encoded_support_audio + support_labels_downsampled
            query_cat = encoded_query_audio + query_masked_labels
        else:
            support_cat = torch.cat((encoded_support_audio, support_labels_downsampled), dim=1) # (batch, embedding_dim+4, time/scale_factor)
            query_cat = torch.cat((encoded_query_audio, query_masked_labels), dim=1) # (batch, embedding_dim+4, time/scale_factor)
        
        transformer_input = torch.cat((support_cat, query_cat), dim = 2)
        transformer_input = rearrange(transformer_input, 'b c t -> b t c')
        
        transformer_output = self.transformer(transformer_input)
        transformer_output = rearrange(transformer_output, 'b t c -> b c t')
        
        query_output = transformer_output[:,:,support_len:] # return only embedding for query
        logits = self.logits_head(query_output).squeeze(1)
        confidences = rearrange(query_output, 'b c t -> b t c')
        
        return  logits, confidences # each output: (batch, query_time/scale_factor). 
        
class AvesEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.aves_embedding = AvesEmbedding(args)
        self.args = args

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
        
        if x.size(1) % self.args.scale_factor != 0:
            raise Exception("audio length is not divisible by scale factor")
            
        expected_dur_output = x.size(1)//self.args.scale_factor
        
        feats = self.aves_embedding(x)
        feats = rearrange(feats, 'b t c -> b c t')

        #embedding may be off by 1 sample from expected
        pad = expected_dur_output - feats.size(2)
        if pad>0:
            feats = F.pad(feats, (0,pad), mode='reflect')
            
        return feats

    def freeze(self):
        self.aves_embedding.freeze()
          
    def unfreeze(self):
        self.aves_embedding.unfreeze()

class ATSTEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.atst = load_model(args.atst_model_path, device=device)
        self.args = args
    
    def forward(self, x):
        encoding = get_timestamp_embedding(x, self.atst)
        return encoding
    
    def freeze(self):
        self.atst.freeze()

    def unfreeze(self):
        self.atst.unfreeze()

class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.atst_frame:
            self.audio_encoder = ATSTEncoder(args)
        else:
            self.audio_encoder = AvesEncoder(args)

        self.label_encoder = LabelEncoder(args)
        self.args = args
        self.audio_chunk_size_samples = int(args.sr * args.audio_chunk_size_sec)
        self.confidence_transformer = ContinuousTransformerWrapper(
            dim_in = 512,
            dim_out = 1,
            max_seq_len = 0,
            use_abs_pos_emb = False,
            attn_layers = Encoder(
                dim = 128,
                depth = 2,
                heads = 2,
                ff_swish = True,
                ff_glu = True,
                attn_flash=True,
            )
        )
        
        
         
        cl = torch.normal(torch.zeros((1,1,512)), torch.ones((1,1,512)))
        self.cls_token = torch.nn.parameter.Parameter(data=cl.to(device))
        # assert self.audio_chunk_size_samples % 2 == 0, "chunk size must be even, to allow for 50% windowing"

    def forward(self, support_audio, support_labels, query_audio, query_labels=None, temperature=1):
        """
        Input
            support_audio (Tensor): (batch, time) (at audio_sr)
            support_labels (Tensor): (batch, time) (at audio_sr)
            query_audio (Tensor): (batch, time) (at audio_sr)
            query_labels (Tensor): (batch, time) (at audio_sr)
        Output
            logits (Tensor): (batch, query_time/scale_factor) (at audio_sr / scale factor)
        """
        # pad and normalize audio
        # support_pad_len = (self.audio_chunk_size_samples - support_audio.size(1) % self.audio_chunk_size_samples) % self.audio_chunk_size_samples
        support_pad_len = self.audio_chunk_size_samples - (support_audio.size(1) % self.audio_chunk_size_samples)
        if support_pad_len>0:
            support_labels = F.pad(support_labels, (0,support_pad_len))

        
        normalization_factor = torch.std(support_audio, dim=1, keepdim=True)
        normalization_factor = torch.maximum(normalization_factor, torch.full_like(normalization_factor, 1e-6))
        support_audio = (support_audio - torch.mean(support_audio, dim=1,keepdim=True)) / normalization_factor
        query_audio = (query_audio - torch.mean(query_audio, dim=1,keepdim=True)) / normalization_factor
        
        # encode audio and labels
        query_logits = []
        query_confidences = []
        
        # with torch.no_grad(): # don't backprop across query embedding, since it is duplicated so much will slow down & cause imbalance
        #     query_audio_encoded = self.audio_encoder(query_audio) # (batch, embedding_dim, time/scale_factor)
        query_audio_encoded = self.audio_encoder(query_audio) # (batch, embedding_dim, time/scale_factor)
            
        support_len_samples = support_audio.size(1)
        for start_sample in range(0, support_len_samples, self.audio_chunk_size_samples):
            support_audio_sub = support_audio[:, start_sample:start_sample+self.audio_chunk_size_samples]
            support_audio_sub_encoded = self.audio_encoder(support_audio_sub)
            
            support_labels_sub = support_labels[:, start_sample:start_sample+self.audio_chunk_size_samples]
            support_labels_sub_downsampled = F.max_pool1d(support_labels_sub.unsqueeze(1), self.args.scale_factor, padding=0).squeeze(1) # (batch, time/scale_factor). 0=NEG 1=UNK 2=POS

            
            l, c = self.label_encoder(support_audio_sub_encoded, support_labels_sub_downsampled, query_audio_encoded) # each output: (batch, query_time/scale_factor). 
            
            c_shape = c.size() # b t c
            
            query_logits.append(l)
            c = torch.reshape(c, (-1, c_shape[2]))
            query_confidences.append(c)
        
        query_confidences = torch.stack(query_confidences, 1) # bt n_support c
        cls_token = self.cls_token.expand(query_confidences.size(0), -1, -1) # bt 1 c
        query_confidences = torch.cat([cls_token, query_confidences], dim=1)
        query_logits = self.confidence_transformer(query_confidences)[:,0,:].squeeze(-1).squeeze(-1) #bt 1 1 -> bt
        query_logits = torch.reshape(query_logits, (c_shape[0], c_shape[1])) # b t
        weighted_average_logits=query_logits
        
        
        
#         query_logits = torch.stack(query_logits, 1)
#         query_confidences = torch.stack(query_confidences, 1) # bt n_support c
        
#         query_confidences = self.confidence_transformer(query_confidences) # bt n_support 1
#         query_confidences = query_confidences.squeeze(2)
#         query_confidences = torch.reshape(query_confidences, (c_shape[0], c_shape[1], -1)) # b t n_support
#         query_confidences = rearrange(query_confidences, 'b t c -> b c t')
        
#         weights = torch.softmax(query_confidences*(1/temperature), dim=1)
#         weighted_average_logits = (query_logits*weights).sum(dim=1) # (batch, query_time/scale_factor)
        
        # downsample query labels, for training
        if query_labels is not None:
            query_labels = torch.unsqueeze(query_labels, 1) # (batch, 1, time)
            query_labels = F.max_pool1d(query_labels, self.args.scale_factor, padding=0) # (batch, 1 , time/scale_factor). 0=NEG 1=UNK 2=POS
            query_labels = torch.squeeze(query_labels, 1) # (batch, time/scale_factor)
        
        return weighted_average_logits, query_labels

    def freeze_audio_encoder(self):
        self.audio_encoder.freeze()
          
    def unfreeze_audio_encoder(self):
        self.audio_encoder.unfreeze()
    