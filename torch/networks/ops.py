import torch
import torch.nn as nn


class MaskingAllocation(nn.Module):
    def __init__(self):
        super(MaskingAllocation, self).__init__()

    def forward(self, w_source, w_target):
        b_size = w_source.shape[0]

        w_target_t = torch.transpose(w_target, 1, 2)

        scores = torch.matmul(w_source, w_target_t)
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
        batch_size, H, W = h.size(0), h.size(2), h.size(3)

        q = self.conv0(h)
        q = q.view(batch_size, -1, H * W)

        k = self.fc_k(words)
        # k = k.transpose(1, 2)

        attn = torch.matmul(k, q)
        attn = attn.view(batch_size, -1, H, W)
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
        beta, gamma = torch.transpose(beta, 1, 2), torch.transpose(gamma, 1, 2)

        B, H, W = attn.size(0), attn.size(2), attn.size(3)

        attn = attn.view(attn.size(0), attn.size(1), -1)
        beta = torch.matmul(beta, attn)
        gamma = torch.matmul(gamma, attn)
        
        beta = beta.view(B, -1, H, W)
        gamma = gamma.view(B, -1, H, W)
        
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
        self.conv_1 = nn.Conv2d(ch_out, ch_out, 3, 1, 1)
        self.macam_1 = MACAM(ch_out, w_dim, q_dim)
        self.l_relu = nn.LeakyReLU(2e-1)
        self.skip_flag = ch_in != ch_out
        if self.skip_flag:
            self.skip_conv = nn.Conv2d(ch_in, ch_out, 1, 1, bias=False)

    def shortcut(self, h):
        if self.up_sample:
            h = self.up(h)
        if self.skip_flag:
            h = self.skip_conv(h)
        return h

    def residual(self, h, w_source, w_target):
        h = self.macam_0(h, w_source, w_target)
        h = self.l_relu(h)
        if self.up_sample:
            h = self.up(h)
        h = self.conv_0(h)
        h = self.macam_1(h, w_source, w_target)
        h = self.l_relu(h)
        h = self.conv_1(h)
        return h

    def forward(self, h, w_source, w_target):
        return self.shortcut(h) + self.residual(h, w_source, w_target)


# ############## D networks ##########################
def downBlock(in_planes, out_planes):
    block = [
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)]
    return block


def Block3x3_leakRelu(in_planes, out_planes):
    block = [
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)]
    return block

def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)