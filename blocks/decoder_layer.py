import torch.nn as nn

from blocks.attention import MultiHeadAttention
from blocks.feedforward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model = d_model, n_heads = n_heads, dropout = dropout)
        self.decoder_attention = MultiHeadAttention(d_model = d_model, n_heads = n_heads, dropout = dropout)

        self.feedforward = FeedForward(d_model = d_model, d_ff = d_ff, dropout = dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None):
        tgt2 = self.self_attention(tgt, tgt, tgt, tgt_mask)
        tgt2 = self.dropout1(tgt2)
        tgt = tgt + tgt2

        tgt2 = self.decoder_attention(tgt, memory, memory, memory_mask)
        tgt2 = self.dropout2(tgt2)
        tgt = tgt + tgt2

