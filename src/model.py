import itertools
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from src.utils import SFeatDataSet, SingleModalityDataSet, masked_bce, log_zinb_positive
from src.module import Exp, Encoder, Decoder, SplitEnc, SplitDec, TestEnc, TestDec, NaiveAffineDiscriminator, AffineTransform


class AutoEncoder(nn.Module):
    '''
    Addapted from https://github.com/wukevin/babel/blob/main/babel/models/autoencoders.py
    '''

    def __init__(
        self,
        input_dim: int,
        z_dim: int = 20,
        activation=nn.PReLU,
        seed=8947,
        output_encoded: bool = True,
        final_activation=[Exp(), nn.Softplus(), nn.Sigmoid()], hiddens=[1], batchnorm=True, n_samples = 1
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.output_encoded = output_encoded
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.n_samples = n_samples
        assert len(final_activation) == 3

        self.encoder = Encoder(
            self.input_dim, z_dim=self.z_dim, activation=activation
        )
        self.decoder = Decoder(
            self.input_dim,
            z_dim=self.z_dim,
            activation=activation,
            final_activation=final_activation
        )

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        if self.output_encoded:
            return encoded, decoded
        return decoded


class SplitAE(nn.Module):
    """
    DeepVAE: FC Variational AutoEncoder
    """

    def __init__(self, input_dim=1, hiddens=[1], z_dim=20, batchnorm=False, n_samples = 1):
        super().__init__()
        self.n_samples = n_samples
        c = 1
        if batchnorm is not False:
            if batchnorm == True:
                if n_samples > 1:
                    c = 1 + self.n_samples
            else:
                c = 1
        else:
            c = 1
        self.enc = SplitEnc(input_dim, z_dim)
        self.dec = SplitDec(input_dim, z_dim+c)

    def encode(self, x, depths = None):
        latent, _ = self.enc(x)
        return latent

    def decode(self, latent, depths = None, batchs = None):
        if batchs is not None:
            latent = torch.cat([latent, depths, batchs], 1)
        elif depths is not None:
            latent = torch.cat([latent, depths], 1)
        recon = self.dec(latent)
        return recon

    def embed(self, x, depths):
        latent = self.encode(x, depths)
        return latent

    def forward(self, x, depths, batchs = None):
        #print(f'input shape: {x.shape}')
        latent = self.encode(x, depths)
        output = self.decode(latent, depths = depths, batchs = batchs)
        return latent, output


class SplitVAE(nn.Module):
    """
    Split Conditional VAE for Multi-chromosome ATAC Embedding
    """

    def __init__(self, input_dim, hiddens=[1], z_dim=20, batchnorm=False, n_samples = 1):
        super(SplitVAE, self).__init__()
        self.n_samples = n_samples
        if batchnorm and self.n_samples > 1:
            c = 1 + self.n_samples
        else:
            c = 1
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.Encoder = SplitEnc(input_dim, z_dim)
        self.Decoder = SplitDec(input_dim, z_dim+c)
        self.exp = Exp()

    def reparam(self, mean, log_var):
        z = Normal(mean, self.exp(log_var)).rsample()
        return z

    def decode(self, mean, log_var, l, b=None, return_rsample=False):
        z = self.reparam(mean, log_var)
        if b is not None:
            z_c = torch.cat((z, l, b), 1)
        else:
            z_c = torch.cat((z, l), 1)

        rec = self.Decoder(z_c)

        if return_rsample:
            return z, rec
        else:
            return rec

    def encode(self, x):
        mean, log_var = self.Encoder(x)
        return mean, log_var

    def embed(self, x, l, b=None):
        mean, _ = self.encode(x)
        return mean

    def forward(self, x, l, b=None, no_rec=False):
        if no_rec:
            mean, log_var = self.encode(x)
            return mean, log_var
        else:
            mean, log_var = self.encode(x)
            z, rec = self.decode(mean, log_var, l, b, return_rsample=True)
            return mean, log_var, z, rec


class TestAE(nn.Module):
    def __init__(self, input_dim, *args, **kwargs):
        super(TestAE, self).__init__()
        input_dim = input_dim
        self.enc = TestEnc(input_dim)
        self.dec = TestDec(input_dim)
        self.kwargs = kwargs

    def encode(self, x):
        latent, _ = self.enc(x)
        return latent

    def embed(self, x):
        return self.encode(x)

    def decode(self, latent):
        recon = self.dec(latent)
        return recon

    def forward(self, x, *args, **kwargs):
        latent = self.encode(x)
        recon = self.decode(latent)
        return latent, recon


class TestVAE(nn.Module):
    """
    Variational Auto-encoder for RNA embedding
    """

    def __init__(self, input_dim, *args, **kwargs):
        super(TestVAE, self).__init__()
        input_dim = input_dim
        self.enc = TestEnc(input_dim)
        self.dec = TestDec(input_dim)
        self.exp = Exp()

    def reparam(self, mean, log_var):
        z = Normal(mean, self.exp(log_var)).rsample()
        return z

    def decode(self, mean, log_var, return_rsample=False):
        z = self.reparam(mean, log_var)
        rec = self.dec(z)

        if return_rsample:
            return z, rec
        else:
            return rec

    def encode(self, x):
        mean, log_var = self.enc(x)
        return mean, log_var

    def embed(self, x):
        mean, _ = self.encode(x)
        return mean

    def forward(self, x, no_rec=False, *args, **kwargs):
        if no_rec:
            mean, log_var = self.encode(x)
            return mean, log_var
        else:
            mean, log_var = self.encode(x)
            z, rec = self.decode(mean, log_var, return_rsample=True)
            return mean, log_var, z, rec


class DeepAE(nn.Module):
    """
    DeepAE: FC AutoEncoder
    """

    def __init__(self, input_dim=1, hiddens=[1], batchnorm=False):
        super(DeepAE, self).__init__()
        self.depth = len(hiddens)
        self.channels = [input_dim] + hiddens  # [5, 3, 3]

        encoder_layers = []
        # for i in range(self.depth - 1):
        for i in range(self.depth):
            encoder_layers.append(
                nn.Linear(self.channels[i], self.channels[i + 1]))
            if i < self.depth - 1:
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                if batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(self.channels[i + 1]))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self.depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self.channels[i], self.channels[i - 1]))
            decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            if i > 1 and batchnorm:
                decoder_layers.append(nn.BatchNorm1d(self.channels[i - 1]))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output


class MultimodalGAN:
    def __init__(self, args, config, atac_ratio=0.05):
        self.args = args
        self.config = config
        self.atac_ratio = atac_ratio
        self.device = config['device']

        # init logger and log all setting params
        self._init_logger()
        self.logger.debug('All settings used:')
        for k, v in sorted(vars(self.args).items()):
            self.logger.debug("{0}: {1}".format(k, v))
        for k, v in sorted(self.config.items()):
            self.logger.debug("{0}: {1}".format(k, v))

        # init tensorboard
        if self.args.task_type == 'train':
            self._init_writer()

        # make dataloader
        self._build_dataloader()

        # calculate total number of iterations
        self.total_iter = 0

        # Generators
        self.latent_dim = config['rna_hiddens']
        self.use_vae = True if self.args.generator_type == 'vae' else False

        if self.use_vae:
            self.rnaAE = TestVAE(
                input_dim=np.sum(self.train_data_rna.chr),
                hiddens=config['rna_hiddens'],
                batchnorm=config['batchnorm'],
                n_samples=config['n_samples']
            )
            self.atacAE = SplitVAE(
                input_dim=self.train_data_atac.chr,
                hiddens=config['atac_hiddens'],
                batchnorm=config['batchnorm'],
                n_samples=config['n_samples']
            )
        else:
            self.rnaAE = TestAE(
                input_dim=np.sum(self.train_data_rna.chr),
                hiddens=config['rna_hiddens'],
                batchnorm=config['batchnorm'],
                n_samples=config['n_samples']
            )
            self.atacAE = SplitAE(
                input_dim=self.train_data_atac.chr,
                hiddens=config['atac_hiddens'],
                batchnorm=config['batchnorm'],
                n_samples=config['n_samples']
            )

        # self.rna2atac = DeepAE(
        #     input_dim=self.latent_dim,
        #     hiddens=config['rna2atac_hiddens'],
        #     batchnorm=config['batchnorm']
        # )
        # self.atac2rna = DeepAE(
        #     input_dim=self.latent_dim,
        #     hiddens=config['atac2rna_hiddens'],
        #     batchnorm=config['batchnorm']
        # )
        # self.rna2atac = PositionwiseFeedForward(self.latent_dim, 64)
        # self.atac2rna = PositionwiseFeedForward(self.latent_dim, 64)
        self.rna2atac = AffineTransform(
            self.latent_dim, 64, affine_num=self.args.affine_num)
        self.atac2rna = AffineTransform(
            self.latent_dim, 64, affine_num=self.args.affine_num)

        # Discriminators (modality classifiers)
        # self.D_rna = nn.Sequential(
        #     nn.Linear(self.latent_dim, int(self.latent_dim / 4)),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(int(self.latent_dim / 4), 1)
        # )
        # self.D_atac = nn.Sequential(
        #     nn.Linear(self.latent_dim, int(self.latent_dim / 4)),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(int(self.latent_dim / 4), 1)
        # )
        self.D_rna = NaiveAffineDiscriminator(self.latent_dim, affine_num=self.args.affine_num)
        self.D_atac = NaiveAffineDiscriminator(self.latent_dim, affine_num=self.args.affine_num)

        # Optimizers
        self.optimizer_G1 = optim.Adam(
            self.rnaAE.parameters(),
            lr=self.args.lr_ae,
            # betas=(self.args.b1, self.args.b2),
            weight_decay=self.args.weight_decay
        )
        self.optimizer_G2 = optim.Adam(
            self.atacAE.parameters(),
            lr=self.args.lr_ae,
            # betas=(self.args.b1, self.args.b2),
            weight_decay=self.args.weight_decay
        )

        # collect all params in generators
        params = [
            {
                'params': itertools.chain(self.rnaAE.parameters(), self.atacAE.parameters()),
                'lr': self.args.lr_ae
            },
            {
                'params': itertools.chain(self.rna2atac.parameters(), self.atac2rna.parameters())
            }
        ]

        if self.args.gan_type == 'wasserstein':
            self.optimizer_D = optim.Adam(
                itertools.chain(self.D_rna.parameters(),
                                self.D_atac.parameters()),
                lr=self.args.lr_d,
                weight_decay=self.args.weight_decay
            )
            self.optimizer_G = optim.Adam(
                params,
                lr=self.args.lr_g,
                weight_decay=self.args.weight_decay
            )
            self.optimizer_G_cross = optim.Adam(
                [params[1]],
                lr=self.args.lr_g,
                weight_decay=self.args.weight_decay
            )
        else:
            self.optimizer_D = optim.Adam(
                itertools.chain(self.D_rna.parameters(),
                                self.D_atac.parameters()),
                lr=self.args.lr_d,
                betas=(self.args.b1, self.args.b2),
                weight_decay=self.args.weight_decay
            )
            self.optimizer_G = optim.Adam(
                params,
                lr=self.args.lr_g,
                betas=(self.args.b1, self.args.b2),
                weight_decay=self.args.weight_decay
            )
            self.optimizer_G_cross = optim.Adam(
                [params[1]],
                lr=self.args.lr_g,
                weight_decay=self.args.weight_decay
            )

        # TODO: not using it at this time
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_G, patience=self.args.scheduler_patience, factor=0.5)

        '''
        freeze rnaAE and atacAE (joint training)

        TODO: the following lines are commented, not sure whether is needed to freeze paramaters here - JL
        '''
        # if self.args.train_type == 'joint':
        #     for param in self.rnaAE.parameters():
        #         param.requires_grad = False

        #     for param in self.atacAE.parameters():
        #         param.requires_grad = False

        # Loss function
        self.adv_loss_fn = F.binary_cross_entropy_with_logits

    def __atac_vae_loss(self, x_atac, z_mean, z_logvar, mean, var, recon, cycle):
        # kl_w = np.min([2 * (self.total_iter - (self.total_iter//cycle) * cycle) / cycle, 1])
        # kld_z = kl(
        #     Normal(z_mean, stable_exp(z_logvar).sqrt()),
        #     Normal(mean, var)
        # ).sum() * kl_w * 2
        kl_w, kld_z = torch.tensor(0.), torch.tensor(0.)

        # pos_weight = torch.Tensor([100]).to(self.device)
        # rec_loss = F.binary_cross_entropy_with_logits(recon, x_atac, reduction='sum', pos_weight=pos_weight) * (1 + self.args.atac_lambda)
        rec_loss = masked_bce(recon, x_atac)

        # rec_loss = regularized_bce(recon, x_atac, omega=self.atac_ratio)
        # m_kld = (apprx_kl(z_mean, self.atacAE.exp(z_logvar).sqrt()).sum() - 0.5 * self.atacAE.z_dim) * kl_w * self.args.atac_lambda
        m_kld = torch.tensor(0.)

        loss = kld_z + rec_loss + m_kld
        output = {
            'loss': loss,
            'log': {
                'vae_loss': loss.item(),
                'kld_z': kld_z.item(),
                'recon': rec_loss.item(),
                'm_kld': m_kld.item(),
                'kl_w': kl_w
            }
        }
        return output

    def __step_atac(self, x_atac, depths_atac, epoch, batch_atac = None, cycle=100):
        atac_feats = x_atac

        # train with variational auto encoder or auto encoder
        if self.use_vae:
            z_atacs_mean, z_atacs_logvar, atacs_latent, atacs_recon = self.atacAE(
                atac_feats, depths_atac
            )
            mean = torch.zeros_like(z_atacs_mean)
            var = torch.ones_like(z_atacs_logvar)

            self.total_iter += 1

            # step loss
            output = self.__atac_vae_loss(
                atac_feats, z_atacs_mean, z_atacs_logvar, mean, var, atacs_recon, cycle)
        else:
            #print(f'batches: {batch_atac}')
            atacs_latent, atacs_recon = self.atacAE(atac_feats, depths_atac, batchs = batch_atac)
            atac_recon_loss = masked_bce(atacs_recon, x_atac)
            output = {
                'loss': atac_recon_loss,
                'log': {
                    'ae_loss': atac_recon_loss.item()
                }
            }

        loss = output['loss']
        return loss, output['log']

    def __rna_vae_loss(self, x_rna, z_mean, z_logvar, mean, var, mu, theta, pi, cycle):
        # kl_w = np.min([2 * (self.total_iter -(self.total_iter//cycle) * cycle) / cycle, 1])
        # kld_z = kl(
        #     Normal(z_mean, stable_exp(z_logvar).sqrt()),
        #     Normal(mean, var)
        # ).sum(-1).mean()
        kld_z = torch.tensor(0.)
        rec_loss = -log_zinb_positive(x_rna, mu, theta, pi).mean()

        # m_kld = apprx_kl(z_mean, torch.exp(z_logvar).sqrt()).mean() - 0.5 * self.atacAE.z_dim
        # 2*kld_z*kl_w + (1+self.args.atac_lambda)*rec_loss + m_kld*kl_w*self.args.atac_lambda
        loss = kld_z + rec_loss
        output = {
            'loss': loss,
            'log': {
                'vae_loss': loss.item(),
                'kld_z': kld_z.item(),
                'recon': rec_loss.item()
                # 'm_kld': m_kld.item()
            }
        }
        return output

    def __step_rna(self, x_rna, epoch, cycle=100):
        rna_feats = x_rna

        # train with variational auto encoder or auto encoder
        if self.use_vae:
            rnas_mean, rnas_logvar, rnas_latent, rnas_recon = self.rnaAE(
                rna_feats)
            mu, theta, pi = rnas_recon
            output = self.__rna_vae_loss(
                x_rna, rnas_mean, rnas_logvar, torch.zeros_like(rnas_mean),
                torch.ones_like(rnas_logvar), mu, theta, pi, cycle
            )
        else:
            rnas_latent, rnas_recon = self.rnaAE(rna_feats)
            mu, theta, pi = rnas_recon
            # rec_loss = -log_zinb_positive(x_rna, mu, theta, pi).mean()
            loss_fn = nn.MSELoss()
            rec_loss = loss_fn(mu, x_rna)
            output = {
                'loss': rec_loss,
                'log': {
                    'ae_loss': rec_loss.item()
                }
            }

        loss = output['loss']
        return loss, output['log']

    def rna_warm_up(self):
        step = 0
        for epoch in range(self.args.warmup_epochs):
            zinb_recorder = []
            for _, (x_rna, depths_rna, _) in enumerate(self.train_loader_rna):
                x_rna = x_rna.to(self.device)
                depths_rna = depths_rna.to(self.device)

                _, _, _, rec = self.rnaAE(x_rna)
                mu, theta, pi = rec
                zinb = -log_zinb_positive(x_rna, mu, theta, pi).mean()
                # rec_loss = focal(rec.view(-1), x.view(-1).long())
                self.optimizer_G1.zero_grad()
                zinb.backward()
                self.optimizer_G1.step()
                zinb_recorder.append(float(zinb))

                # log loss
                if (step + 1) % self.args.log_freq == 0:
                    self.writer.add_scalar(
                        'Warmup_rna/loss', zinb.item(), step)

                # increase step
                step += 1
            self.logger.info(
                f'RNA Warm up epoch {epoch}: rec loss = {np.average(zinb_recorder)}')

    def atac_warm_up(self):
        step = 0
        for epoch in range(self.args.warmup_epochs):
            rec_loss_recorder = []
            for _, (x_atac, depths_atac, _) in enumerate(self.train_loader_atac):
                x_atac = x_atac.to(self.device)
                depths_atac = depths_atac.to(self.device)

                _, _, _, rec = self.atacAE(x_atac, depths_atac)
                # rec_loss = regularized_bce(rec, x_atac, omega=self.atac_ratio)
                # rec_loss = focal(rec.view(-1), x.view(-1).long())
                rec_loss = masked_bce(rec, x_atac)
                self.optimizer_G2.zero_grad()
                rec_loss.backward()
                self.optimizer_G2.step()
                rec_loss_recorder.append(float(rec_loss))

                # log loss
                if (step + 1) % self.args.log_freq == 0:
                    self.writer.add_scalar(
                        'Warmup_atac/loss', rec_loss.item(), step)

                # increase step
                step += 1

            self.logger.info(
                f'ATAC Warm up epoch {epoch}: rec loss = {np.average(rec_loss_recorder)}')

    def __step_joint_vae(self, x_rna, x_atac, depths_atac, step, epoch, cycle):
        rna_feats = x_rna
        atac_feats = x_atac

        rna_batch_size = x_rna.size()[0]
        atac_batch_size = x_atac.size()[0]

        # ADDED
        # Train Generators
        self.optimizer_G.zero_grad()
        rna_loss, rna_log_dict = self.__step_rna(x_rna, epoch)
        atac_loss, atac_log_dict = self.__step_atac(
            x_atac, depths_atac, epoch, cycle=cycle)
        vae_gen_loss = rna_loss + atac_loss
        vae_gen_loss.backward()
        self.optimizer_G.step()
        self.optimizer_G.zero_grad()

        # END
        # Train cycle consitency
        # fix the latent embedding modules
        with torch.no_grad():
            z_rna_mean, z_rna_logvar = self.rnaAE(rna_feats, no_rec=True)
            z_atac_mean, z_atac_logvar = self.atacAE(
                atac_feats, depths_atac, no_rec=True)

        atac_latent = z_atac_mean
        rna_latent = z_rna_mean

        rna2atac_latent = self.rna2atac(rna_latent)
        rna_latent_recon = self.atac2rna(rna2atac_latent)

        atac2rna_latent = self.atac2rna(atac_latent)
        atac_latent_recon = self.rna2atac(atac2rna_latent)

        # latent cycle loss
        rna_latent_cycle_loss = F.l1_loss(rna_latent, rna_latent_recon)
        atac_latent_cycle_loss = F.l1_loss(atac_latent, atac_latent_recon)

        # feature cycle loss (reconstruction)
        mu, theta, pi = self.rnaAE.decode(rna_latent_recon, z_rna_logvar)
        rna_recon_cycle_loss = self.__rna_vae_loss(
            x_rna, z_rna_mean, z_rna_logvar, torch.zeros_like(
                z_rna_mean), torch.ones_like(z_rna_logvar), mu, theta, pi, 100
        )['loss']

        atac_recon = self.atacAE.decode(
            atac_latent_recon, z_atac_logvar, depths_atac)
        atac_recon_cycle_loss = self.__atac_vae_loss(
            atac_feats, z_atac_mean, z_atac_logvar, torch.zeros_like(
                z_atac_mean),
            torch.ones_like(z_atac_logvar), atac_recon, 100
        )['loss']

        # Total recon loss (cycle)
        recon_loss = self.args.lamda1 * (rna_latent_cycle_loss + atac_latent_cycle_loss) +\
            (rna_recon_cycle_loss + atac_recon_cycle_loss)

        # ----------------
        # Train Generator
        # ----------------
        rna_real = torch.ones(rna_batch_size, 1).to(self.device)
        rna_fake = torch.zeros(rna_batch_size, 1).to(self.device)
        atac_real = torch.ones(atac_batch_size, 1).to(self.device)
        atac_fake = torch.zeros(atac_batch_size, 1).to(self.device)

        if self.args.gan_type == 'naive':
            d_loss = self.adv_loss_fn(self.D_rna(atac2rna_latent), atac_real) +\
                self.adv_loss_fn(self.D_atac(rna2atac_latent), rna_real)
        elif self.args.gan_type == 'wasserstein':
            d_loss = -self.D_rna(rna_latent, atac2rna_latent).mean() - \
                self.D_atac(atac_latent, rna2atac_latent).mean()

        # only update cross mapping layers
        G_loss = recon_loss + self.args.lamda3 * d_loss
        self.optimizer_G_cross.zero_grad()
        G_loss.backward()
        self.optimizer_G_cross.step()

        # prepare log dict
        log_dict = {}
        log_dict['recon'] = vae_gen_loss.item()
        log_dict['cross_recon'] = recon_loss.item()
        log_dict['d_loss'] = d_loss.item()
        log_dict['G_loss'] = G_loss.item()

        sub_log_dict = {}
        for k, v in atac_log_dict.items():
            sub_log_dict[f'vae_atac_{k}'] = v
        for k, v in rna_log_dict.items():
            sub_log_dict[f'vae_rna_{k}'] = v
        sub_log_dict['rna_latent_cycle_loss'] = rna_latent_cycle_loss.item()
        sub_log_dict['atac_latent_cycle_loss'] = atac_latent_cycle_loss.item()
        sub_log_dict['rna_recon_cycle_loss'] = rna_recon_cycle_loss.item()
        sub_log_dict['atac_recon_cycle_loss'] = atac_recon_cycle_loss.item()

        # -------------------
        # Train Discriminator
        # -------------------
        if (step + 1) % self.args.update_d_freq == 0:
            # sample a new batch data to be the same space guess
            dis_x_rna, _ = self._sample_one_batch_from_loader(
                'train_dis_loader_rna')
            dis_x_atac, dis_depths_atac = self._sample_one_batch_from_loader(
                'train_dis_loader_atac')
            dis_x_rna = dis_x_rna.to(self.device)
            dis_x_atac = dis_x_atac.to(self.device)
            dis_depths_atac = dis_depths_atac.to(self.device)

            with torch.no_grad():
                dis_rna_latent, _ = self.rnaAE(dis_x_rna, no_rec=True)
                dis_atac_latent, _ = self.atacAE(
                    dis_x_atac, dis_depths_atac, no_rec=True)

            if self.args.gan_type == 'naive':
                D_rna_loss = (self.adv_loss_fn(self.D_rna(rna_latent.detach()), rna_real) +
                              self.adv_loss_fn(self.D_rna(atac2rna_latent.detach()), atac_fake)) / 2
                D_atac_loss = (self.adv_loss_fn(self.D_atac(atac_latent.detach()), atac_real) +
                               self.adv_loss_fn(self.D_atac(rna2atac_latent.detach()), rna_fake)) / 2
            elif self.args.gan_type == 'wasserstein':
                # we hope same space groups are somehow agree with each other, while different space groups are disagree
                D_rna_loss = self.D_rna(rna_latent.detach(), atac2rna_latent.detach()).mean() -\
                    self.D_rna(rna_latent.detach(), dis_rna_latent).mean()
                D_atac_loss = self.D_atac(atac_latent.detach(), rna2atac_latent.detach()).mean() -\
                    self.D_atac(atac_latent.detach(), dis_atac_latent).mean()

            D_loss = (D_rna_loss + D_atac_loss) * self.args.lamda3
            self.optimizer_D.zero_grad()
            D_loss.backward()
            self.optimizer_D.step()

            # weight clipping
            if self.args.gan_type == 'wasserstein':
                for p in self.D_rna.parameters():
                    p.data.clamp_(-self.args.weight_clip_value,
                                  self.args.weight_clip_value)

                for p in self.D_atac.parameters():
                    p.data.clamp_(-self.args.weight_clip_value,
                                  self.args.weight_clip_value)

            # logging
            log_dict['D_loss'] = D_loss.item()
        return log_dict, sub_log_dict

    def __step_joint_ae(self, x_rna, x_atac, depths_atac, step, epoch, cycle):
        rna_feats = x_rna
        atac_feats = x_atac

        rna_batch_size = x_rna.size()[0]
        atac_batch_size = x_atac.size()[0]

        # ADDED
        # Train Generators
        self.optimizer_G.zero_grad()
        rna_loss, rna_log_dict = self.__step_rna(x_rna, epoch)
        atac_loss, atac_log_dict = self.__step_atac(
            x_atac, depths_atac, epoch, cycle=cycle)
        vae_gen_loss = rna_loss + atac_loss
        vae_gen_loss.backward()
        self.optimizer_G.step()
        self.optimizer_G.zero_grad()

        # END
        # Train cycle consitency
        # fix the latent embedding modules
        with torch.no_grad():
            z_rna_mean, _ = self.rnaAE(rna_feats)
            z_atac_mean, _ = self.atacAE(atac_feats, depths_atac)

        atac_latent = z_atac_mean
        rna_latent = z_rna_mean

        rna2atac_latent = self.rna2atac(rna_latent)
        rna_latent_recon = self.atac2rna(rna2atac_latent)

        atac2rna_latent = self.atac2rna(atac_latent)
        atac_latent_recon = self.rna2atac(atac2rna_latent)

        # latent cycle loss
        rna_latent_cycle_loss = F.l1_loss(rna_latent, rna_latent_recon)
        atac_latent_cycle_loss = F.l1_loss(atac_latent, atac_latent_recon)

        # feature cycle loss (reconstruction)
        mu, theta, pi = self.rnaAE.decode(rna_latent_recon)
        rna_recon_cycle_loss = -log_zinb_positive(x_rna, mu, theta, pi).mean()

        atac_recon = self.atacAE.decode(atac_latent_recon)
        atac_recon_cycle_loss = masked_bce(atac_recon, x_atac)

        # Total recon loss (cycle)
        recon_loss = self.args.lamda1 * (rna_latent_cycle_loss + atac_latent_cycle_loss) +\
            (rna_recon_cycle_loss + atac_recon_cycle_loss)

        # ----------------
        # Train Generator
        # ----------------
        rna_real = torch.ones(rna_batch_size, 1).to(self.device)
        rna_fake = torch.zeros(rna_batch_size, 1).to(self.device)
        atac_real = torch.ones(atac_batch_size, 1).to(self.device)
        atac_fake = torch.zeros(atac_batch_size, 1).to(self.device)

        if self.args.gan_type == 'naive':
            d_loss = self.adv_loss_fn(self.D_rna(atac2rna_latent), atac_real) +\
                self.adv_loss_fn(self.D_atac(rna2atac_latent), rna_real)
        elif self.args.gan_type == 'wasserstein':
            d_loss = -self.D_rna(rna_latent, atac2rna_latent).mean() - \
                self.D_atac(atac_latent, rna2atac_latent).mean()

        # only update cross mapping layers
        G_loss = recon_loss + self.args.lamda3 * d_loss
        self.optimizer_G_cross.zero_grad()
        G_loss.backward()
        self.optimizer_G_cross.step()

        # prepare log dict
        log_dict = {}
        log_dict['recon'] = vae_gen_loss.item()
        log_dict['cross_recon'] = recon_loss.item()
        log_dict['d_loss'] = d_loss.item()
        log_dict['G_loss'] = G_loss.item()

        sub_log_dict = {}
        for k, v in atac_log_dict.items():
            sub_log_dict[f'ae_atac_{k}'] = v
        for k, v in rna_log_dict.items():
            sub_log_dict[f'ae_rna_{k}'] = v
        sub_log_dict['rna_latent_cycle_loss'] = rna_latent_cycle_loss.item()
        sub_log_dict['atac_latent_cycle_loss'] = atac_latent_cycle_loss.item()
        sub_log_dict['rna_recon_cycle_loss'] = rna_recon_cycle_loss.item()
        sub_log_dict['atac_recon_cycle_loss'] = atac_recon_cycle_loss.item()

        # -------------------
        # Train Discriminator
        # -------------------
        if (step + 1) % self.args.update_d_freq == 0:
            # sample a new batch data to be the same space guess
            dis_x_rna, _ = self._sample_one_batch_from_loader(
                'train_dis_loader_rna')
            dis_x_atac, dis_depths_atac = self._sample_one_batch_from_loader(
                'train_dis_loader_atac')
            dis_x_rna = dis_x_rna.to(self.device)
            dis_x_atac = dis_x_atac.to(self.device)
            dis_depths_atac = dis_depths_atac.to(self.device)

            with torch.no_grad():
                dis_rna_latent, _ = self.rnaAE(dis_x_rna)
                dis_atac_latent, _ = self.atacAE(dis_x_atac, dis_depths_atac)

            if self.args.gan_type == 'naive':
                D_rna_loss = (self.adv_loss_fn(self.D_rna(rna_latent.detach()), rna_real) +
                              self.adv_loss_fn(self.D_rna(atac2rna_latent.detach()), atac_fake)) / 2
                D_atac_loss = (self.adv_loss_fn(self.D_atac(atac_latent.detach()), atac_real) +
                               self.adv_loss_fn(self.D_atac(rna2atac_latent.detach()), rna_fake)) / 2
            elif self.args.gan_type == 'wasserstein':
                # we hope same space groups are somehow agree with each other, while different space groups are disagree
                D_rna_loss = self.D_rna(rna_latent.detach(), atac2rna_latent.detach()).mean() -\
                    self.D_rna(rna_latent.detach(), dis_rna_latent).mean()
                D_atac_loss = self.D_atac(atac_latent.detach(), rna2atac_latent.detach()).mean() -\
                    self.D_atac(atac_latent.detach(), dis_atac_latent).mean()

            D_loss = (D_rna_loss + D_atac_loss) * self.args.lamda3
            self.optimizer_D.zero_grad()
            D_loss.backward()
            self.optimizer_D.step()

            # weight clipping
            if self.args.gan_type == 'wasserstein':
                for p in self.D_rna.parameters():
                    p.data.clamp_(-self.args.weight_clip_value,
                                  self.args.weight_clip_value)

                for p in self.D_atac.parameters():
                    p.data.clamp_(-self.args.weight_clip_value,
                                  self.args.weight_clip_value)

            # logging
            log_dict['D_loss'] = D_loss.item()
        return log_dict, sub_log_dict

    def __step_joint_pseudo(
        self, x_rna, x_atac, depths_atac, samples_atac, x_rna_idx, x_atac_idx, step
    ):
        rna_feats = x_rna
        atac_feats = x_atac
        rna_batch_size = x_rna.size()[0]
        atac_batch_size = x_atac.size()[0]

        # END
        # Train cycle consitency
        # fix the latent embedding modules
        z_rna_mean, _ = self.rnaAE(rna_feats)
        z_atac_mean, _ = self.atacAE(atac_feats, depths_atac, batchs = samples_atac)

        atac_latent = z_atac_mean
        rna_latent = z_rna_mean
        rna_pseudo_label = self.label_array[x_rna_idx.flatten()]
        atac_pseudo_label = self.pseudo_label_array[x_atac_idx.flatten()]

        rna2atac_latent = self.rna2atac(rna_latent)
        rna_latent_recon = self.atac2rna(rna2atac_latent)

        atac2rna_latent = self.atac2rna(atac_latent)
        atac_latent_recon = self.rna2atac(atac2rna_latent)
        
        # latent cycle loss
        rna_latent_cycle_loss = F.l1_loss(rna_latent, rna_latent_recon)
        atac_latent_cycle_loss = F.l1_loss(atac_latent, atac_latent_recon)

        # feature cycle loss (reconstruction)
        mu, theta, pi = self.rnaAE.decode(rna_latent_recon)
        # rna_recon_cycle_loss = -log_zinb_positive(x_rna, mu, theta, pi).mean()
        rna_recon_cycle_loss_fn = nn.MSELoss()
        rna_recon_cycle_loss = rna_recon_cycle_loss_fn(mu, x_rna)
        #rna_recon_cycle_loss = nn.functional.mse_loss(x_rna, mu)#-log_zinb_positive(x_rna, mu, theta, pi).mean()

        atac_recon = self.atacAE.decode(atac_latent_recon, depths = depths_atac, batchs = samples_atac)
        atac_recon_cycle_loss = masked_bce(atac_recon, x_atac)

        # Total recon loss (cycle)
        recon_loss = self.args.lamda1 * (rna_latent_cycle_loss + atac_latent_cycle_loss) +\
            (rna_recon_cycle_loss + atac_recon_cycle_loss)

        # Penalty for closing to any centroid
        # centroid_loss = centroids_penalty(atac2rna_latent, rna_centroids) + centroids_penalty(rna2atac_latent, atac_centroids)

        # discriminator loss
        rna_real = torch.ones(rna_batch_size, 1).to(self.device)
        rna_fake = torch.zeros(rna_batch_size, 1).to(self.device)
        atac_real = torch.ones(atac_batch_size, 1).to(self.device)
        atac_fake = torch.zeros(atac_batch_size, 1).to(self.device)
        
        if self.args.gan_type == 'naive':
            d_loss = self.adv_loss_fn(self.D_rna(atac2rna_latent, atac_pseudo_label), atac_real) +\
                self.adv_loss_fn(self.D_atac(rna2atac_latent, rna_pseudo_label), rna_real) +\
                self.adv_loss_fn(self.D_rna(rna_latent_recon, rna_pseudo_label), atac_real) +\
                self.adv_loss_fn(self.D_atac(atac_latent_recon, atac_pseudo_label), rna_real)
        elif self.args.gan_type == 'wasserstein':
            d_loss = -(self.D_rna(atac2rna_latent).max(-1)[0] + self.D_atac(rna2atac_latent).max(-1)[0] + self.D_rna(rna_latent_recon, rna_pseudo_label) + self.D_atac(atac_latent_recon, atac_pseudo_label)).mean()

        # only update cross mapping layers
        G_loss = recon_loss + self.args.lamda3 * d_loss
        self.optimizer_G_cross.zero_grad()
        G_loss.backward()
        self.optimizer_G_cross.step()

        # prepare log dict
        log_dict = {}
        log_dict['cross_recon'] = recon_loss.item()
        # log_dict['centroids_penalty'] = centroid_loss.item()
        log_dict['d_loss'] = d_loss.item()
        log_dict['G_loss'] = G_loss.item()

        sub_log_dict = {}
        sub_log_dict['rna_latent_cycle_loss'] = rna_latent_cycle_loss.item()
        sub_log_dict['atac_latent_cycle_loss'] = atac_latent_cycle_loss.item()
        sub_log_dict['rna_recon_cycle_loss'] = rna_recon_cycle_loss.item()
        sub_log_dict['atac_recon_cycle_loss'] = atac_recon_cycle_loss.item()

        if (step + 1) % self.args.update_d_freq == 0:
            # sample a new batch data to be the same space guess
            # dis_x_rna, _ = self._sample_one_batch_from_loader(
            #     'train_dis_loader_rna')
            # dis_x_atac, dis_depths_atac = self._sample_one_batch_from_loader(
            #     'train_dis_loader_atac')
            # dis_x_rna = dis_x_rna.to(self.device)
            # dis_x_atac = dis_x_atac.to(self.device)
            # dis_depths_atac = dis_depths_atac.to(self.device)

            # with torch.no_grad():
            #     dis_rna_latent, _ = self.rnaAE(dis_x_rna)
            #     dis_atac_latent, _ = self.atacAE(dis_x_atac, dis_depths_atac)

            if self.args.gan_type == 'naive':
                D_rna_loss = (self.adv_loss_fn(self.D_rna(rna_latent.detach(), rna_pseudo_label), rna_real) +
                              self.adv_loss_fn(self.D_rna(atac2rna_latent.detach(), rna_pseudo_label), atac_fake)) / 2
                D_atac_loss = (self.adv_loss_fn(self.D_atac(atac_latent.detach(), atac_pseudo_label), atac_real) +
                               self.adv_loss_fn(self.D_atac(rna2atac_latent.detach(), atac_pseudo_label), rna_fake)) / 2
            elif self.args.gan_type == 'wasserstein':
                # we hope same space groups are somehow agree with each other, while different space groups are disagree
                D_rna_loss = (self.D_rna(atac2rna_latent.detach()) - self.D_rna(rna_latent.detach(), rna_pseudo_label)).mean()
                D_atac_loss = (self.D_atac(rna2atac_latent.detach()) - self.D_atac(atac_latent.detach(), atac_pseudo_label)).mean()

            D_loss = (D_rna_loss + D_atac_loss) * self.args.lamda3
            self.optimizer_D.zero_grad()
            D_loss.backward()
            self.optimizer_D.step()

            # weight clipping
            if self.args.gan_type == 'wasserstein':
                for p in self.D_rna.parameters():
                    p.data.clamp_(-self.args.weight_clip_value,
                                  self.args.weight_clip_value)

                for p in self.D_atac.parameters():
                    p.data.clamp_(-self.args.weight_clip_value,
                                  self.args.weight_clip_value)

            # logging
            log_dict['D_loss'] = D_loss.item()
        return log_dict, sub_log_dict

    def train(self, epoch, train_type='rna'):
        """Train model with different training strategies (i.e. rna, atac, joint, joint_pseudo).
        """
        self.set_model_status(training=True)

        # compute cycle number and iteration number
        cycle = 100 * self.train_loader_atac.__len__() // self.args.batch_size  # // self.cuda_dev
        # total_iter = (epoch - 1) * \
        #     self.train_loader_atac.__len__() // self.args.batch_size + 1

        # loss variables
        loss = 0
        loss_G = 0
        loss_D = 0
        log_dict = {}
        sub_log_dict = {}

        start_time = time.time()
        for step, ((x_rna, depths_rna, x_rna_idx), (x_atac, depths_atac, x_atac_idx)) in enumerate(zip(self.train_loader_rna, self.train_loader_atac)):
            is_multisample = os.path.exists(os.path.join(self.args.data_dir, 'samples.txt'))
            if is_multisample:
                depths_atac, samples_atac = depths_atac
                #print(f'size of depths: {depths_atac.shape}')
                #print(f'size of samples: {samples_atac.shape}')
            if train_type == 'rna':
                self.optimizer_G1.zero_grad()
                x_rna = x_rna.to(self.device)
                #depths_rna = depths_rna.to(self.device)
                rna_loss, log_dict = self.__step_rna(x_rna, epoch)
                rna_loss.backward()
                self.optimizer_G1.step()
                loss += rna_loss.item()

            if train_type == 'atac':
                self.optimizer_G2.zero_grad()
                x_atac = x_atac.to(self.device)
                depths_atac = depths_atac.to(self.device)
                if is_multisample:
                    samples_atac = samples_atac.to(self.device)
                else:
                    samples_atac = None
                atac_loss, log_dict = self.__step_atac(
                    x_atac, depths_atac, epoch, batch_atac = samples_atac, cycle=cycle)
                atac_loss.backward()
                self.optimizer_G2.step()
                loss += atac_loss.item()

            if train_type == 'joint':
                x_rna = x_rna.to(self.device)
                x_atac = x_atac.to(self.device)
                depths_atac = depths_atac.to(self.device)

                if self.use_vae:
                    log_dict, sub_log_dict = self.__step_joint_vae(
                        x_rna, x_atac, depths_atac, step, epoch, cycle
                    )
                else:
                    log_dict, sub_log_dict = self.__step_joint_ae(
                        x_rna, x_atac, depths_atac, step, epoch, cycle
                    )
                loss_G += log_dict.get('G_loss', 0.)
                loss_D += log_dict.get('D_loss', 0.)

            if train_type == 'joint_pseudo':
                x_rna = x_rna.to(self.device)
                x_atac = x_atac.to(self.device)
                x_rna_idx = x_rna_idx.to(self.device)
                x_atac_idx = x_atac_idx.to(self.device)
                samples_atac = None#samples_atac.to(self.device)
                depths_atac = depths_atac.to(self.device)

                log_dict, sub_log_dict = self.__step_joint_pseudo(
                    x_rna, x_atac, depths_atac, samples_atac, x_rna_idx, x_atac_idx,
                    step
                )
                loss_G += log_dict.get('G_loss', 0.)
                loss_D += log_dict.get('D_loss', 0.)

            # -------------------
            # UPDATE LOGGINGS
            # -------------------
            if (step + 1) % self.args.log_freq == 0:
                assert self.args.log_freq % self.args.update_d_freq == 0
                info_string = ' | '.join(
                    [f'{k}: {v:.4f}' for k, v in log_dict.items()])
                self.logger.info("EPOCH: {} | ITER:{}/{} | {} | Time: {:.2f}s".format(
                    epoch, step, len(self.train_loader_atac), info_string, time.time() - start_time))
                for k, v in log_dict.items():
                    self.writer.add_scalar(
                        f'Train_{train_type}/{k}', v,
                        step + len(self.train_loader_atac) * epoch
                    )

                for k, v in sub_log_dict.items():
                    self.writer.add_scalar(
                        f'Train_{train_type}_details/{k}', v,
                        step + len(self.train_loader_atac) * epoch
                    )

                start_time = time.time()

        # -------------------
        # SAVE CHECKPOINT
        # -------------------
        if epoch > self.args.save_ratio * self.args.n_epochs and (epoch + 1) % self.args.save_freq == 0:
            self.save_cpt(epoch)

        if self.args.train_type in ['joint', 'joint_pseudo']:
            for group_id, param_group in enumerate(self.optimizer_G.param_groups):
                self.writer.add_scalar(
                    f'Train_{train_type}_lr/group_{group_id}', param_group['lr'], epoch)
            return {'epoch_loss_G': loss_G, 'epoch_loss_D': loss_D}
        else:
            return {'epoch_loss': loss}

    def embedding(self, dataloader, source_modal='rna'):
        self.set_model_status(training=False)
        with torch.no_grad():
            rna, atac = [], []

            for step, (x, depths, _) in enumerate(dataloader):
                x = x.to(self.device)
                if os.path.exists(os.path.join(self.args.data_dir, 'samples.txt')):
                    depths, samples = depths
                    samples = samples.to(self.device)
                depths = depths.to(self.device)
                
                

                if source_modal == 'rna':
                    latent = self.rnaAE.embed(x)
                    rna_latent = latent
                    rna.append(rna_latent.cpu().numpy())

                    atac_latent = self.rna2atac(rna_latent)
                    atac.append(atac_latent.cpu().numpy())
                else:
                    latent = self.atacAE.embed(x, depths)

                    atac_latent = latent
                    atac.append(atac_latent.cpu().numpy())

                    rna_latent = self.atac2rna(atac_latent)
                    rna.append(rna_latent.cpu().numpy())

            rna = np.vstack(rna)
            atac = np.vstack(atac)

            return rna, atac
    
    def predict_rna_from_atac(self, dataloader):
        self.set_model_status(training=False)
        with torch.no_grad():
            res = []

            for step, (x, depths, _) in enumerate(dataloader):
                x = x.to(self.device)
                if os.path.exists(os.path.join(self.args.data_dir, 'samples.txt')):
                    depths, samples = depths
                    samples = samples.to(self.device)
                depths = depths.to(self.device)

                latent = self.atacAE.embed(x, depths)
                
                recon = self.atac2rna(latent)
                
                mu, theta, pi = self.rnaAE.decode(recon)

                res.append(mu.cpu().numpy())

            res = np.vstack(res)

            return res
        
    def predict_atac_from_rna(self, dataloader, depth = 6000, n_samples = 5):
        self.set_model_status(training=False)
        with torch.no_grad():
            res = []

            for step, (x, depths, _) in enumerate(dataloader):
                x = x.to(self.device)
                if os.path.exists(os.path.join(self.args.data_dir, 'samples.txt')):
                    depths, samples = depths
                    samples = samples.to(self.device)
                depths = depths.to(self.device)

                latent = self.rnaAE.embed(x)
                
                recon = self.rna2atac(latent)
                
                mu = self.atacAE.decode(recon, depths = (torch.ones(x.shape[0], 1) * depth).to(self.device), batchs = (torch.ones(x.shape[0], n_samples) / n_samples).to(self.device))

                res.append(mu.cpu().numpy())

            res = np.vstack(res)

            return res

    def _build_dataloader(self):
        # define dataloader params
        kwargs = {'num_workers': self.args.n_cpu}

        # make rna dataloader
        rna_chr_fpath = os.path.join(self.args.data_dir,
                                      'train_rna_chromosome.txt')
        sample_fpath = os.path.join(self.args.data_dir, 'samples.txt')

        if not os.path.exists(sample_fpath):
            sample_fpath = None
        if not os.path.exists(rna_chr_fpath):
            rna_chr_fpath = None

        self.train_data_rna = SingleModalityDataSet(
            file_mat=os.path.join(self.args.data_dir, 'train_rna.npz'),
            file_chr=rna_chr_fpath,
            file_sample = sample_fpath
        )

        self.train_loader_rna = DataLoader(
            dataset=self.train_data_rna,
            batch_size=self.args.batch_size,
            shuffle=True,
            **kwargs
        )

        self.train_dis_loader_rna = DataLoader(
            dataset=self.train_data_rna,
            batch_size=self.args.batch_size,
            shuffle=True,
            **kwargs
        )

        # TODO: we use training data as proxy testing data here
        #self.test_data_rna = SingleModalityDataSet(
        #    file_mat=os.path.join(self.args.data_dir, 'train_rna.npz'),
        #    file_chr=os.path.join(self.args.data_dir,
        #                          'train_rna_chromosome.txt')
        #)
        self.test_loader_rna = DataLoader(
            dataset=self.train_data_rna,
            batch_size=self.args.batch_size,
            shuffle=False,
            **kwargs
        )

        # make atac dataloader
        self.train_data_atac = SingleModalityDataSet(
            file_mat=os.path.join(self.args.data_dir, 'train_atac.npz'),
            file_chr=os.path.join(self.args.data_dir,
                                  'train_atac_chromosome.txt'), 
            file_sample = sample_fpath
        )
        self.train_loader_atac = DataLoader(
            dataset=self.train_data_atac,
            batch_size=self.args.batch_size,
            shuffle=True,
            **kwargs
        )
        self.train_dis_loader_atac = DataLoader(
            dataset=self.train_data_atac,
            batch_size=self.args.batch_size,
            shuffle=True,
            **kwargs
        )

        # TODO: we use training data as proxy testing data here
        #self.test_data_atac = SingleModalityDataSet(
        #    file_mat=os.path.join(self.args.data_dir, 'train_atac.npz'),
        #    file_chr=os.path.join(self.args.data_dir,
        #                          'train_atac_chromosome.txt')
        #)
        self.test_loader_atac = DataLoader(
            dataset=self.train_data_atac,
            batch_size=self.args.batch_size,
            shuffle=False,
            **kwargs
        )

        # build label mapping
        label_frame = pd.read_csv(self.args.embedding_label_path, delimiter='\t', header=None)
        labels = label_frame.iloc[:, 1].tolist()

        pseudo_label_frame = pd.read_csv(self.args.embedding_pseudo_label_path, delimiter='\t', header=None)
        pseudo_labels = pseudo_label_frame.iloc[:, 1].tolist()

        # prepare label array
        label_encoder = LabelEncoder()
        self.label_array = torch.from_numpy(label_encoder.fit_transform(labels)).to(self.device)
        self.pseudo_label_array = torch.from_numpy(label_encoder.transform(pseudo_labels)).to(self.device)

    def _build_pretrain_dataloader(self, modal='rna'):
        kwargs = {'num_workers': self.args.n_cpu, 'pin_memory': True}
        train_modal_data = SFeatDataSet(
            file_mat=os.path.join(self.args.data_dir,
                                  'train_{}.mat'.format(modal)))
        train_modal_loader = DataLoader(dataset=train_modal_data,
                                        batch_size=self.args.batch_size,
                                        shuffle=True, **kwargs)
        return train_modal_loader

    def _sample_one_batch_from_loader(self, dataloader_name):
        try:
            dataloader_iter = getattr(self, f'{dataloader_name}_iter')
            data = next(dataloader_iter)
        except:
            dataloader = getattr(self, dataloader_name)
            setattr(self, f'{dataloader_name}_iter', iter(dataloader))
            dataloader_iter = getattr(self, f'{dataloader_name}_iter')
            data = next(dataloader_iter)
        return data

    def _get_pseudo_cluster(self, source='rna'):
        cluster_name = f'_{source}_pseudo_cluster'

        if hasattr(self, cluster_name):
            cluster = getattr(self, cluster_name)
        else:
            self.logger.info(f'Runing KMeans on {source} embedding...')
            if source == 'rna':
                embedding = self.embedding(
                    self.train_loader_rna,
                    source_modal='rna'
                )[0]
            elif source == 'atac':
                embedding = self.embedding(
                    self.train_loader_atac,
                    source_modal='atac'
                )[1]

            cluster = KMeans(
                n_clusters=self.args.affine_num,
                random_state=self.args.seed
            )
            cluster.fit(embedding)
            setattr(self, cluster_name, cluster)

        return cluster

    def set_model_status(self, training=True):
        """Set all modules states

        Args:
            training (bool, optional): training state. Defaults to True.
        """
        if training:
            self.rnaAE.train()
            self.atacAE.train()
            self.rna2atac.train()
            self.atac2rna.train()
            self.D_rna.train()
            self.D_atac.train()
        else:
            self.rnaAE.eval()
            self.atacAE.eval()
            self.rna2atac.eval()
            self.atac2rna.eval()
            self.D_rna.eval()
            self.D_atac.eval()

    def to_cuda(self):
        self.rnaAE.cuda()
        self.atacAE.cuda()
        self.rna2atac.cuda()
        self.atac2rna.cuda()
        self.D_rna.cuda()
        self.D_atac.cuda()

    def get_state(self, epoch):
        state_dict = {'epoch': epoch,
                      'G1_state_dict': self.rnaAE.state_dict(),
                      'G2_state_dict': self.atacAE.state_dict(),
                      'G12_state_dict': self.rna2atac.state_dict(),
                      'G21_state_dict': self.atac2rna.state_dict(),
                      'D1_state_dict': self.D_rna.state_dict(),
                      'D2_state_dict': self.D_atac.state_dict(),
                      'optimizer_G': self.optimizer_G.state_dict(),
                      'optimizer_D': self.optimizer_D.state_dict(),
                      'scheduler': self.scheduler.state_dict()
                      }
        return state_dict

    def save_cpt(self, epoch):
        state_dict = self.get_state(epoch)
        cptname = '{}_checkpt_{}.pkl'.format(
            self.args.train_type, epoch)   # original name: 'rna_atac'
        cptpath = os.path.join(self.args.cpt_dir, cptname)
        self.logger.info("> Save checkpoint '{}'".format(cptpath))
        torch.save(state_dict, cptpath)

    def load_cpt(self, cptpath):
        if os.path.isfile(cptpath):
            self.logger.info("> Load checkpoint '{}'".format(cptpath))
            dicts = torch.load(cptpath)
            self.epoch = dicts['epoch']
            self.rnaAE.load_state_dict(dicts['G1_state_dict'])
            self.atacAE.load_state_dict(dicts['G2_state_dict'])
            self.rna2atac.load_state_dict(dicts['G12_state_dict'])
            self.atac2rna.load_state_dict(dicts['G21_state_dict'])
            self.D_rna.load_state_dict(dicts['D1_state_dict'])
            self.D_atac.load_state_dict(dicts['D2_state_dict'])
            self.optimizer_G.load_state_dict(dicts['optimizer_G'])
            self.optimizer_D.load_state_dict(dicts['optimizer_D'])
            self.scheduler.load_state_dict(dicts['scheduler'])
        else:
            self.logger.error("> No checkpoint found at '{}'".format(cptpath))

    def load_pretrain_cpt(self, cptpath, modal='rna', only_weight=False):
        if os.path.isfile(cptpath):
            self.logger.info("Load checkpoint '{}'".format(cptpath))
            dicts = torch.load(cptpath)
            if modal == 'rna':
                self.rnaAE.load_state_dict(dicts['G1_state_dict'])
                optimizer = self.optimizer_G1
            elif modal == 'atac':
                self.atacAE.load_state_dict(dicts['G2_state_dict'])
                optimizer = self.optimizer_G2

            if not only_weight:
                self.epoch = dicts['epoch']
                optimizer.load_state_dict(dicts['optimizer'])
        else:
            self.logger.error("No checkpoint found at '{}'".format(cptpath))

    def _init_writer(self):
        self.logger.info('> Create writer at \'{}\''.format(self.args.cpt_dir))
        self.writer = SummaryWriter(self.args.cpt_dir)

    def _init_logger(self):
        logging.basicConfig(
            filename=os.path.join(self.args.cpt_dir, self.config['log_file']),
            level=logging.INFO,
            datefmt='%Y/%m/%d %H:%M:%S',
            format='%(asctime)s: %(name)s [%(levelname)s] %(message)s'
        )
        formatter = logging.Formatter(
            '%(asctime)s: %(name)s [%(levelname)s] %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S'
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(stream_handler)
