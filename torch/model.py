import torch
import torch.nn as nn

from networks.encoder import ResNetbdcnEncoder
from networks.text_encoder import EncoderText
from networks.generator import Generator
from networks.discriminator import Discriminator


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self._build_model()
        self._build_dataloader()
        self._build_optimizer()

    def _build_model(self):
        self.image_enc = ResNetbdcnEncoder(self.opt.edge_level, self.opt.edge_model_path, self.opt.edge_tanh)
        self.text_enc = EncoderText(self.opt)
        self.netG = Generator(
            self.opt.w_dim, self.opt.q_dim, self.opt.img_size)
        self.netD = Discriminator(
            self.opt.img_size, self.opt.ch_in, self.opt.embedding_dim)

    def _build_dataloader(self):
        self.dataset = None

    def _build_optimizer(self):
        G_params = list(self.netG.parameters())
        D_params = list(self.netD.parameters())

        beta1, beta2 = self.opt.beta1, self.opt.beta2
        G_lr, D_lr = self.opt.lr, self.opt.lr

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D

    def train(self):
        for i in range(self.total_epochs):
            for data in self.dataset:
                g_loss = self.compute_g_loss(data)
                g_loss.backward()
                self.optimizer_G.step()

                d_loss = self.compute_d_loss(data)
                d_loss.backward()
                self.optimizer_D.step()

    def compute_g_loss(self, data):
        img1, img2, source_cap, intra_cap, inter_cap = data

        img1_emb = self.image_enc(img1)
        source_w_embs, source_s_emb = self.text_enc(source_cap)
        intra_w_embs, intra_s_emb = self.text_enc(intra_cap)
        inter_w_embs, inter_s_emb = self.text_enc(inter_cap)

        intra_fake_img = self.netG(img1_emb, source_w_embs, intra_w_embs)
        inter_fake_img = self.netG(img1_emb, source_w_embs, inter_w_embs)
        recon_img = self.netG(img1_emb, source_w_embs, source_w_embs)

        img1_feature = self.netD.feature(img1)
        intra_fake_feature = self.netD.feature(intra_fake_img)
        inter_fake_feature = self.netD.feature(inter_fake_img)

        # adversarial loss
        logits_intra = self.netD.adv(inter_fake_feature)
        logtis_inter = self.netD.adv(intra_fake_feature)
        adv_loss = self.cross_entropy_from_logits(logits_intra, 1) + self.cross_entropy_from_logits(logtis_inter, 1)

        # matching loss
        match_intra = self.netD.match(img1_feature, intra_fake_feature, source_s_emb - intra_s_emb)
        match_inter = self.netD.match(img1_feature, inter_fake_feature, source_s_emb - inter_s_emb)
        matching_loss = self.cross_entropy_from_logits(match_intra, 1) + self.cross_entropy_from_logits(match_inter, 1)

        # reconstruction loss
        recon_loss = nn.L1Loss(recon_img, img1)
        
        g_loss = adv_loss + matching_loss + recon_loss

        return g_loss

    def compute_d_loss(self, data):
        img1, img2, source_cap, intra_cap, inter_cap = data

        img1_emb = self.image_enc(img1)
        img2_emb = self.image_enc(img2)
        source_w_embs, source_s_emb = self.text_enc(source_cap)
        intra_w_embs, intra_s_emb = self.text_enc(intra_cap)
        inter_w_embs, inter_s_emb = self.text_enc(inter_cap)

        intra_fake_img = self.netG(img1_emb, source_w_embs, intra_w_embs)
        inter_fake_img = self.netG(img1_emb, source_w_embs, inter_w_embs)

        img1_feature = self.netD.feature(img1)
        img2_feature = self.netD.feature(img2)
        intra_fake_feature = self.netD.feature(intra_fake_img)
        inter_fake_feature = self.netD.feature(inter_fake_img)

        # adversarial loss
        logits_intra = self.netD.adv(inter_fake_feature)
        logtis_inter = self.netD.adv(intra_fake_feature)
        logtis_real1 = self.netD.adv(img1_feature)
        logtis_real2 = self.netD.adv(img2_feature)
        adv_loss = self.cross_entropy_from_logits(logits_intra, 0) + self.cross_entropy_from_logits(logtis_inter, 0) \
            + self.cross_entropy_from_logits(logtis_real1, 1) + self.cross_entropy_from_logits(logtis_real2, 1)

        # matching loss
        match_intra = self.netD.match(img1_feature, intra_fake_feature, source_s_emb - intra_s_emb)
        match_inter = self.netD.match(img1_feature, inter_fake_feature, source_s_emb - inter_s_emb)
        match_real1 = self.netD.match(img1_feature, img1_feature, source_s_emb - intra_s_emb)
        match_real2 = self.netD.match(img1_feature, img2_feature, source_s_emb - inter_s_emb)
        matching_loss = self.cross_entropy_from_logits(match_intra, 0) + self.cross_entropy_from_logits(match_inter, 0) \
            + self.cross_entropy_from_logits(match_real1, 1) + self.cross_entropy_from_logits(match_real2, 1)

        d_loss = adv_loss + matching_loss

        return d_loss

    def cross_entropy_from_logits(logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target)
        entropy = F.binary_cross_entropy_with_logits(logits, targets)
        return entropy
