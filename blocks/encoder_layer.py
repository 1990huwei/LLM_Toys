import torch.nn as nn

from blocks.attention import MultiHeadAttention
from blocks.feedforward import FeedForward

class EncoderLayerPostLN(nn.Module):
    def __init__(self, d_model, n_heads, d_ff = 2048, dropout = 0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model = d_model, n_heads=n_heads, dropout = dropout)
        self.feed_forward = FeedForward(d_model = d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask = None):
        src2 = self.multi_head_attention(src, src, src, mask = src_mask)
        src2 = self.dropout1(src2)
        src = src + src2
        src = self.norm1(src)

        src2 = self.feed_forward(src)
        src2 = self.dropout2(src2)
        src = src + src2
        src = self.norm2(src)

        return src


class EncoderLayerPreLN(nn.Module):
    def __init__(self, d_model, n_heads, d_ff = 2048, dropout = 0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model = d_model, n_heads=n_heads, dropout = dropout)
        self.feed_forward = FeedForward(d_model = d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask = None):
        src2 = self.norm1(src)
        src2 = self.multi_head_attention(src2, src2, src2, mask = src_mask)
        src2 = self.dropout1(src2)
        src = src + src2

        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        src2 = self.dropout2(src2)
        src = src + src2
        return src