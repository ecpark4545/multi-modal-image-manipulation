import torch
import torch.nn as nn
import math


class AdaIN(nn.Module):
    def __init__(self, d_style, channels_in, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.channels_in = channels_in
        self.epsilon = epsilon
        self.norm = nn.InstanceNorm2d(channels_in, affine=False)
        self.fc = nn.Linear(d_style, channels_in * 2, bias=True)
        
    def forward(self, x, style):
        h = self.fc(style)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class ResBlk(nn.Module):
    def __init__(self):
        

    def short_cut(self, x):

    def residual(self, x):
        
    def forward(self, x):


class AdaINResBlock(nn.Module):
    def __init__(self, channels_in, channels_out, up_sample=False):
        super(AdaINResBlock, self).__init__()
        
        
    def shortcut(self, x):
        if self.up_sample

    def residual(self, x):
    
    def forward(self, x, style):
        x = self.residual(x, style) + self.shortcut(x)
        return x / math.sqrt(2)