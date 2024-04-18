import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention
from feedforward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff = 2048, dropout = 0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model = d_model, n_heads=n_heads, dropout = dropout)
        self.feed_forward = FeedForward(d_model = d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask = None):
        attention_output = self.multi_head_attention(src, src, src, mask = src_mask)
        attention_output = self.dropout1(attention_output)
        src = src + attention_output
        src = self.norm1(src)

        feedforward_output = self.feed_forward(src)
        feedforward_output = self.dropout2(feedforward_output)
        src = src + feedforward_output
        src = self.norm2(src)

        return src
