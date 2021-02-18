import torch
import torch.nn as nn
import numpy as np

from ops import *


class Decoder(nn.Module):
    def __init__(self, w_dim, q_dim, img_size=256, img_ch=3, channels=1024):
        super(Decoder, self).__init__()
        self.w_dim = w_dim
        self.q_dim = q_dim
        self.img_size = img_size
        self.img_ch = img_ch
        self.channels = channels

        self.repeat_num = int(np.log2(self.img_size)) - 4
        self._build_model()

    def _build_model(self):
        self.blocks = nn.ModuleList()

        ch_in = self.channels

        # first head block and second of middle block
        for i in range(2):
            self.blocks.append(MACAMResBlock(
                ch_in, ch_in, self.w_dim, self.q_dim))

        # first of middel block with up sampling
        self.blocks.insert(1, MACAMResBlock(
            ch_in, ch_in, self.w_dim, self.q_dim, up_sample=True))

        # up sampling blocks
        for i in range(self.repeat_num):
            ch_out = max(ch_in // 2, 1)
            self.blocks.append(MACAMResBlock(
                ch_in, ch_out, self.w_dim, self.q_dim, up_sample=True))
            ch_in = ch_out

        # to rgb block
        self.to_rgb = nn.Conv2d(ch_out, self.img_ch, 3, 1, 1)

    def forward(self, h, w_source, w_target):
        for block in self.blocks:
            h = block(h, w_source, w_target)
        x = self.to_rgb(h)
        x = torch.F.tanh(x)
        return x
