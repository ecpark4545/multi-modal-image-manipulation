import torch
import torch.nn as nn

import numpy as np

from ops import Block3x3_leakRelu


class D_Feature(nn.Module):
    def __init__(self, img_size, img_ch=3, ch_in=64, max_conv_dim=2048):
        super(D_Feature, self).__init__()
        self.img_size = img_size
        self.img_ch = img_ch
        self.ch_in = ch_in
        self.max_conv_dim = max_conv_dim
        self.num_layers = np.log2(self.img_size) - 6
        self.from_rgb = nn.Conv2d(self.img_ch, self.ch_in, 4, 2, 1, bias=False)
        self.layers = self._build_layers()
        
    def _build_layers(self):
        layers = [nn.LeakyReLU(0.2, inplace=True)]
        ch_in = self.ch_in
        ch_out = self.ch_in
        for i in range(3):
            ch_out = min(ch_in*2, self.max_conv_dim)
            layers += [
                nn.Conv2d(ch_in, ch_out, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True)]
            ch_in = ch_out

        for i in range(self.num_layers):
            ch_out = max(ch_in // 2, 1)
            layers += Block3x3_leakRelu(ch_in, ch_out)
            ch_in = ch_out
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.layers(x)
        return x


class D_Condition(nn.Module):
    def __init__(self, ch_in=64, embedding_dim=256):
        super(D_Condition, self).__init__()
        self.ch_in = ch_in
        self.embedding_dim = embedding_dim
        self.joint_conv = Block3x3_leakRelu(self.ch_in * 8 + self.embedding_dim, self.ch_in)
        self.outlogits = nn.Sequential(
            nn.Conv2d(self.ch_in * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x, condition):
        condition = condition.view(-1, self.embedding_dim, 1, 1)
        condition = condition.repeat(1,1,4,4)
        x_c = torch.cat((x, condition), 1)
        x_c = self.joint_conv(x_c)
        logit = self.outlogits(x_c)
        return logit.view(-1)


class D_Adversarial(nn.Module):
    def __init__(self, ch_in=64):
        super(D_Adversarial, self).__init__()
        self.ch_in = ch_in
        self.outlogits = nn.Sequential(
            nn.Conv2d(self.ch_in * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())
        
    def forward(self, x):
        logit = self.outlogits(x)
        return logit.view(-1)


class D_Match(nn.Module):
    def __init__(self, ch_in=64, embedding_dim=256):
        super(D_MATCH, self).__init__()
        self.ch_in = ch_in
        self.embedding_dim = embedding_dim

        self.outlogits = nn.Sequential(
            nn.Conv2d(n_in_c, 2048, kernel_size=1, padding=0, stride=1), 
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            nn.Conv2d(2048, 1, kernel_size=1, padding=0, stride=1),
            nn.Sigmoid())
    
    def forward(self, x1, x2, s):
        s = s.view(s.size(0), s.size(1), 1, 1)
        s = s.repeat(1, 1, x1.size(2), x1.size(3))
        h = torch.cat([x1, x2, s], dim=1)
        logits = self.outlogits(h)
        return logits.view(-1)


class Discriminator():
    def __init__(self, img_size, ch_in=64, embedding_dim=256):
        self.img_size = img_size
        self.ch_in = ch_in
        self.embedding_dim = embedding_dim
        self.D_feature = D_Feature(self.img_size, ch_in=self.ch_in)
        self.D_condition = D_Condition(ch_in=self.ch_in, embedding_dim=self.embedding_dim)
        self.D_match = D_Match(ch_in=self.ch_in, embedding_dim=self.embedding_dim)
        self.D_adv = D_Adversarial(ch_in=self.ch_in)

