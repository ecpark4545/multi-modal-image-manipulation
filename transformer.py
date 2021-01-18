import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v):
        self.w_q = nn.Linear(d_model, n_head*d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head*d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head*d_v, bias=False)

    def forward(self, q, k, v):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        
