import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head * n_heads == d_model

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self,query, keys, values, mask = None, padding_mask = None):
        batch_size = query.shape[0]
        d_head = self.d_head
        n_heads = self.n_heads

        query = self.q(query).reshape(batch_size, -1, n_heads, d_head)
        keys = self.k(keys).reshape(batch_size, -1, n_heads, d_head)
        values = self.v(values).reshape(batch_size, n_heads, d_head)

        score = torch.einsum('bqnd,bknd->bnqk', query, keys)

        if mask:
            score = torch.masked_fill(score, mask = mask, value = -1e9)
        if padding_mask:
            #padding_mask shape: (batch,seq_q)
            #score shape: (batch,n_heads,seq_q,seq_k)
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            score = torch.masked_fill(score, mask = padding_mask, value = -1e9)

        attention = torch.softmax(score / (d_head ** 0.5), dim = -1)
        attention = self.dropout(attention)