import torch 
import torch.nn as nn

from ops import *

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, x, w_source, w_target):
        return x