import torch
import torch.nn as nn


class MaskingAllocation(nn.Module):
    def __init__(self):
        super(MaskingAllocation, self).__init__()

    def forward(self, w_source, w_target):
        b_size = w_source.shape[0]

        w_target_t = torch.transpose(w_target, 1, 2)

        scores = torch.dot(w_source, w_target_t)
        sum = torch.sum(scores, 1, keepdim=True)
        left_scores = sum - scores
        indices = torch.argmax(left_scores, -1)

        w_target_chunks = torch.chunk(w_target, b_size, 0)
        indice_chunks = torch.chunk(indices, b_size, 0)

        for i in range(b_size):
            index = indice_chunks[i].view(indice_chunks[i].shape[-1])
            if i == 0:
                result = torch.index_select(w_target_chunks[i], 1, index)
            else:
                result = torch.cat(
                    (result, torch.index_select(w_target_chunks[i], 1, index)))
        return result


class CrossAttention(nn.Module):
    def __init__(self, ch_in, w_dim, q_dim):
        super(CrossAttention, self).__init__()
        self.conv0 = nn.Conv2d(ch_in, q_dim, 1, 1, 0)
        self.fc_k = nn.Linear(w_dim, q_dim)

    def forward(self, words, h):
        batch_size, h, w = h.size(0), h.size(1), h.size(2)

        q = self.conv0(h).view(batch_size, h * w, -1)
        k = self.fc_k(words)

        k = k.transpose(1, 2)
        attn = torch.matmul(q, k)
        attn = attn.view(batch_size, h, w, -1)
        return attn


class MACAM(nn.Module):
    def __init__(self, ch_in, w_dim, q_dim):
        super(MACAM, self).__init__()
        self.masking_allocation = MaskingAllocation()
        self.cross_attention = CrossAttention(ch_in, w_dim, q_dim)
        self.fc = nn.Linear(w_dim, ch_in * 2)
        self.instance_norm = nn.InstanceNorm2d(ch_in, affine=True)

    def forward(self, h, w_source, w_target):
        attn = self.cross_attention(w_source, h)
        w_alloc = self.masking_allocation(w_source, w_target)

        beta, gamma = self.fc(w_alloc).chunk(2, -1)

        beta = torch.matmul(attn, beta)
        gamma = torch.matmul(attn, gamma)

        h = self.instance_norm(h) * gamma + beta
        return h


class MACAMResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, w_dim, q_dim, up_sample=False):
        super(MACAMResBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.up_sample = up_sample
        if self.up_sample:
            self.up = nn.Upsample(scale_factor=2)

        self.conv_0 = nn.Conv2d(ch_in, ch_out, 3, 1, 1)
        self.macam_0 = MACAM(ch_in, w_dim, q_dim)
        self.conv_1 = nn.Conv2d(ch_in, ch_out, 3, 1, 1)
        self.macam_1 = MACAM(ch_out, w_dim, q_dim)
        self.l_relu = nn.LeakyReLU(2e-1)
        self.skip_flag = ch_in != ch_out
        if self.skip_flag:
            self.skip_conv = nn.Conv2d(ch_in, ch_out, 1, 1, bias=False)

    def shortcut(self, h):
        if self.up_sample:
            h = self.up(h, scale_factor=2)
        if self.skip_flag:
            h = self.skip_conv(h)
        return h

    def residual(self, h, w_source, w_target):
        h = self.macam_0(h, w_source, w_target)
        h = self.l_relu(h)
        if self.up_sample:
            h = self.up(h, scale_factor=2)
        h = self.conv_0(h)
        h = self.macam_1(h, w_source, w_target)
        h = self.l_relu(h)
        h = self.conv_1(h)
        return h

    def forward(self, h, w_source, w_target):
        return self.shortcut(h) + self.residual(h, w_source, w_target)
