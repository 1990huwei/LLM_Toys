import math
import torch
import torch.nn as nn

class PositionEncoder(nn.Module):
    def __init__(self, d_model = 512, dropout = 0.1, max_seq_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))