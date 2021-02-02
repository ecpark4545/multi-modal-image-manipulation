import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from miscc.config import cfg
from attention import SpatialAttentionGeneral as SPATIAL_ATT
from attention import ChannelAttention as CHANNEL_ATT
from attention import DCMChannelAttention as DCM_CHANNEL_ATT

from ops import *


class CA_NET(nn.Module):
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf + cfg.TEXT.EMBEDDING_DIM  

        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code, cnn_code):

        c_z_code = torch.cat((c_code, z_code), 1)

        # for testing
        if not cfg.TRAIN.FLAG and not cfg.B_VALIDATION:
            cnn_code = cnn_code.repeat(c_z_code.size(0), 1)

        c_z_cnn_code = torch.cat((c_z_code, cnn_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_cnn_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        return out_code64


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = cfg.GAN.R_NUM
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.att = SPATIAL_ATT(ngf, self.ef_dim)            # spatial attention
        self.channel_att = CHANNEL_ATT(ngf, self.ef_dim)    # channel-wise attention
        self.residual = self._make_layer(ResBlock, ngf * 3)
        self.upsample = upBlock(ngf * 3, ngf)
        self.SAIN = ACM(ngf * 3)

    def forward(self, h_code, c_code, word_embs, mask, img):
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        c_code_channel, att_channel = self.channel_att(c_code, word_embs, h_code.size(2), h_code.size(3))
        c_code = c_code.view(word_embs.size(0), -1, h_code.size(2), h_code.size(3))

        h_c_code = torch.cat((h_code, c_code), 1)
        h_c_c_code = torch.cat((h_c_code, c_code_channel), 1)
        h_c_c_img_code = self.SAIN(h_c_c_code, img)

        out_code = self.residual(h_c_c_img_code)
        out_code = self.upsample(out_code)

        return out_code, att


class G_Branch(nn.Module):
    def __init(self, gf_dim, embedding_dim, condition_dim, branch_step, img_ch=3):
        super(G_Branch, self).__init__()
        self.gf_dim = gf_dim
        self.embedding_dim = embedding_dim
        self.condition_dim = condition_dim
        self.branch_step = branch_step
        self.img_ch = img_ch
        self._build_layers()
    
    def _build_layers(self):
        if self.branch_step == 1:
            self.g_stage = INIT_STAGE_G(self.gf_dim * 16, self.condition_dim)
            self.upblock = upBlock(self.embedding_dim, self.gf_dim, scale=3.8)
        else :
            self.g_stage = NEXT_STAGE_G(self.gf_dim, self.embedding_dim, self.condition_dim)
            self.upblock = upBlock(self.gf_dim, self.gf_dim, scale=2)
        self.acm = layers.append(ACM(self.gf_dim))
        self.to_rgb = nn.Sequential(
            nn.Conv2d(self.gf_dim, self.img_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh())
    
    def forward(self, h_code, c_code, word_embs, img_code, mask=None):
        if self.branch_step == 1:
            # initial step of g : h_code <= z_code,  word_embs <= cnn_code,  img_code <= region_features
            h_code = self.g_stage(h_code, c_code, word_embs)
            attn = None
        else :
            h_code, attn = self.g_stage(h_code, c_code, word_embs, mask, img_code)
        up_img_code = self.upblock(img_code)
        h_code_img = self.acm(h_code, up_img_code)
        fake_img = self.to_rgb(h_code_img)
        return h_code, up_img_code, attn, fake_img


class Generator(nn.Module):
    def __init__(self, gf_dim, embedding_dim, condition_dim, branch_num):
        super(Generator, self).__init__()
        self.gf_dim = gf_dim
        self.embedding_dim = embedding_dim
        self.condition_dim = condition_dim
        self.branch_num = branch_num
        self.ca_net = CA_NET()
        self.branches = self._build_branches()

    def _build_branches(self):
        branches = []
        for i in range(self.branch_num):
            branches.append(G_Branch(self.gf_dim, self.embedding_dim, self.condition_dim, branch_step=i))
        return branches

    def forward(self, z_code, sent_emb, word_embs, mask, cnn_code, region_features):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
        """
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb) 
        
        for i in range(self.branch_num):
            if i == 0:
                h_code, img_code, attn, fake_img = self.branches[i](z_code, c_code, cnn_code, region_features)
            else:
                h_code, img_code, attn, fake_img = self.branches[i](h_code, c_code, word_embs, img_code, mask)
            if attn is not None:
                att_maps.append(attn)
            fake_imgs.append(fake_img)

        # The output "h_code3(=h_code)" and "c_code" are used in the DCM
        return fake_imgs, att_maps, mu, logvar, h_code, c_code


class DCM_NEXT_STAGE(nn.Module):
    def __init__(self, ngf, nef, ncf):
        super(DCM_NEXT_STAGE, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = cfg.GAN.R_NUM
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.att = SPATIAL_ATT(ngf, self.ef_dim)
        self.color_channel_att = DCM_CHANNEL_ATT(ngf, self.ef_dim)
        self.residual = self._make_layer(ResBlock, ngf * 3)

        self.block = nn.Sequential(
            conv3x3(ngf * 3, ngf * 2),
            nn.InstanceNorm2d(ngf * 2),
            GLU())

        self.SAIN = ACM(ngf * 3)

    def forward(self, h_code, c_code, word_embs, mask, img):

        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        c_code_channel, att_channel = self.color_channel_att(c_code, word_embs, h_code.size(2), h_code.size(3))
        c_code = c_code.view(word_embs.size(0), -1, h_code.size(2), h_code.size(3))

        h_c_code = torch.cat((h_code, c_code), 1)
        h_c_c_code = torch.cat((h_c_code, c_code_channel), 1)
        h_c_c_img_code = self.SAIN(h_c_c_code, img)

        out_code = self.residual(h_c_c_img_code)
        out_code = self.block(out_code)

        return out_code


# the DCM (detail correction module)
class DCM_Net(nn.Module):
    def __init__(self):
        super(DCM_Net, self).__init__()
        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ncf = cfg.GAN.CONDITION_DIM      
        # ngf, nef, ncf: 32 256 100
        self.img_net = GET_IMAGE_G(ngf)
        self.h_net = DCM_NEXT_STAGE(ngf, nef, ncf)
        self.SAIN = ACM(ngf)
        self.upsample = upBlock(nef//2, ngf)

    def forward(self, x, real_features, sent_emb, word_embs, mask, c_code):

        r_code = self.upsample(real_features)
        h_a_code = self.h_net(x, c_code, word_embs, mask, r_code)
        h_a_r_code = self.SAIN(h_a_code, r_code)
        fake_img = self.img_net(h_a_r_code)

        return fake_img