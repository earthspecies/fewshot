from torch.nn import functional as F
import torch.hub
import math
from einops import rearrange
from torchaudio.models import wav2vec2_model
import json
from torch import nn

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