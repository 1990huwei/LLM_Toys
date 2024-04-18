import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from blocks.encoder_layer import EncoderLayer
from blocks.decoder_layer import DecoderLayer
from blocks.feedforward import FeedForward
from blocks.position_encoder import PositionEncoder

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
        activation: str = "relu",
    ):
        super().__init__()

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.position_encoding = PositionEncoder(d_model, dropout=dropout, max_seq_len=max_seq_len)

        encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)