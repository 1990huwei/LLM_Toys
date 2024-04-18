import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention
from feedforward import FeedForward

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff = 2048, dropout = 0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model = d_model, n_heads=n_heads, dropout = dropout)
        self.feed_forward = FeedForward(d_model = d_model, d_ff=d_ff, dropout=dropout)