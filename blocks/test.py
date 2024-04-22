import math
import torch
import torch.nn as nn

class PositionEmbedding(nn.Module):
    def __init__(self, d_model = 512, max_seq_len = 4096):
        