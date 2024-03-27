import torch
from torch import nn
import torch.hub
from einops import rearrange

from x_transformers import ContinuousTransformerWrapper, Encoder

class AdapterNetwork:
    def __init__(self, args):
        super().__init__()
        self.support_seq_len = int(args.support_dur_sec * args.sr / args.scale_factor)
        self.query_seq_len = int(args.query_dur_sec * args.sr / args.scale_factor)
        assert args.support_dur_sec * args.sr / args.scale_factor == self.support_seq_len
        assert args.query_dur_sec * args.sr / args.scale_factor == self.query_seq_len

        self.transformer = ContinuousTransformerWrapper(
            dim_in = args.embedding_dim,
            dim_out = 512,
            max_seq_len = int(self.support_seq_len + self.query_seq_len),
            attn_layers = Encoder(
                dim = args.label_encoder_dim,
                depth = args.label_encoder_depth,
                heads = args.label_encoder_heads
            )
        )
        self.linear = nn.Linear(512, args.llm_embedding_dim)
        self.args = args

    def forward(self, encoded_support_audio,  encoded_query_audio):

        transformer_input = torch.cat((encoded_support_audio, encoded_query_audio), dim = 2)
        transformer_input = rearrange(transformer_input, 'b c t -> b t c')

        transformer_output = self.transformer(transformer_input)
        transformer_output = rearrange(transformer_output, 'b t c -> b c t')
        return transformer_output #combined embeddings
