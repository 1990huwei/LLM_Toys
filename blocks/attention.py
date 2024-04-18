import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.d_head = d_model // n_heads
        assert(self.d_head * n_heads == d_model)

        self.v = nn.Linear(self.d_head, self.d_head, bias=False)
        self.k = nn.Linear(self.d_head, self.d_head, bias=False)
        self.q = nn.Linear(self.d_head, self.d_head, bias=False)

        self.fc_out = nn.Linear(self.n_heads * self.d_head, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, values, keys, query, mask=None):
        batch_size = values.shape[0]
        d_head = self.d_head
        n_heads = self.n_heads

        values = values.reshape(batch_size, -1, n_heads, d_head)
        keys = keys.reshape(batch_size, -1, n_heads, d_head)
        query = query.reshape(batch_size, -1, n_heads, d_head)

        values = self.v(values)
        keys = self.k(keys)
        query = self.q(query)

        score = torch.einsum('bqnd,bknd->bnqk', query, keys)
        if mask:
            score = score.masked_fill(mask == 0, -1e20)
        
        attention = torch.softmax(score / (self.d_head ** 0.5), dim = -1)
        attention = self.dropout(attention)

        out = torch.einsum('bnqk,bknd->bnqd', attention, values).reshape(batch_size, -1, n_heads * d_head)

        out = self.fc_out(out)
        return out

