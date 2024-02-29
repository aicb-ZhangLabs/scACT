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
from src.model import SplitAE, SplitVAE, TestAE, TestVAE, MultimodalGAN

class RNA2ATAC(nn.Module):
    def __init__(self, args, config, atac_ratio = 0.05, use_cuda = True):
        super(RNA2ATAC, self).__init__()
        self.args = args
        self.config = config
        self.atac_ratio = atac_ratio
        self.model = MultimodalGAN(args, config, atac_ratio = atac_ratio)
        if use_cuda:
            self.model.to_cuda()

    @classmethod
    def from_model(cls, model, use_cuda = True):
        args = model.args
        config = model.config
        atac_ratio = model.atac_ratio
        obj = cls(args, config, atac_ratio = atac_ratio, use_cuda = use_cuda)
        obj.model = model
        return obj

    def load_cpt(self, fpath):
        self.model.load_cpt(fpath)

    def forward(self, X):
        depth = 5000
        n_samples = self.config['n_samples']
        rna_latent = self.model.rnaAE.embed(X)
        recon = self.model.rna2atac(rna_latent)
        mu = self.model.atacAE.decode(recon, depths = (torch.ones(x.shape[0], 1) * depth).to(self.device), batchs = (torch.ones(x.shape[0], n_samples) / n_samples).to(self.device))
        return mu

class ATAC2RNA(nn.Module):
    def __init__(self, args, config, atac_ratio = 0.05, use_cuda = True):
        super(ATAC2RNA, self).__init__()
        self.args = args
        self.config = config
        self.atac_ratio = atac_ratio
        self.model = MultimodalGAN(args, config, atac_ratio = atac_ratio)
        if use_cuda:
            self.model.to_cuda()

    @classmethod
    def from_model(cls, model, use_cuda = True):
        args = model.args
        config = model.config
        atac_ratio = model.atac_ratio
        obj = cls(args, config, atac_ratio = atac_ratio, use_cuda = use_cuda)
        obj.model = model
        return obj

    def load_cpt(self, fpath):
        self.model.load_cpt(fpath)

    def forward(self, X):
        atac_latent = self.model.atacAE.encode(X)
        recon = self.model.atac2rna(atac_latent)
        mu, _, _ = self.model.rnaAE.decode(recon)
        return mu