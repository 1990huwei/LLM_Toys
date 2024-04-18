import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from blocks.encoder_layer import EncoderLayer
from blocks.decoder_layer import DecoderLayer
from blocks.feedforward import FeedForward
from blocks.position_encoder import PositionEncoder


class Encoder(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout, n_layers) -> None:
        super().__init__()
        layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
    
    def forward(self, src, mask = None):
        for layer in self.layers:
            src = layer(src, src_mask = mask)
        return src


class Decoder(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout, n_layers) -> None:
        super().__init__()
        layer = DecoderLayer(d_model, n_heads, d_ff, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
    
    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask = tgt_mask, memory_mask = memory_mask)
        return tgt


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ):
        super().__init__()

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.position_encoding = PositionEncoder(d_model, dropout=dropout, max_seq_len=max_seq_len)

        self.encoder = Encoder(d_model, n_heads, d_ff, dropout, num_encoder_layers)
        self.decoder = Decoder(d_model, n_heads, d_ff, dropout, num_decoder_layers)

        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask = None, tgt_mask = None, memory_mask = None):
        src = self.src_embed(src)
        tgt = self.tgt_embed(tgt)

        src = self.position_encoding(src)
        tgt = self.position_encoding(tgt)

        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        output = self.out(output)

        return output
        