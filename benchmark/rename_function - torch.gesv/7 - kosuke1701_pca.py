#coding: UTF-8

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as _Variable

from log1exp import log1exp

from hybridkg import Variable, Cuda, LVM

class PCA(LVM):
    def _init__(self, n_entity, dim_latent, with_kg=True):
        super(PCA, self).__init__(n_entity, False)

        #NOTE: entity will be associated with each row of loading matrix W
        dim_data = n_entity

        self.dim_latent = dim_latent
        self.dim_data = dim_data
        self.with_kg = with_kg

        #NOTE: self.W is transposed version of W in equation y~WX+mu
        self.W = nn.Parameter(
            torch.FloatTensor(np.random.uniform(-1.,1.,size=(dim_latent, dim_data))/dim_latent**0.5)
        )
        self.mu = nn.Parameter(
            torch.FloatTensor(np.zeros(dim_data))
        )
        self.log_sigma2 = nn.Parameter(
            torch.FloatTensor(np.ones(1))
        )

        if with_kg:
            self.e = nn.Parameter(
                torch.FloatTensor(np.random.normal(size=(n_entity, dim_emb)))
            )
            self.f_e_to_w = nn.Linear(dim_emb, dim_latent)
            self.log_V2 = nn.Parameter(
                torch.FloatTensor(np.ones(1))
            )

    def _encode(self, idx):
        if not self.with_kg:
            raise Exception("No KG mode.")
        if self.if_reparam:
            raise NotImplementedError("hoge")

        return self.e[idx]

    def loss_e(self, idx):
        u"""
            calculate loss w.r.t. embeddings of data entities.
            idx -- entity indices of entities in mini-batch
        """
        if self.if_reparam:
            raise NotImplementedError("hoge")

        if not isinstance(idx, _Variable):
            idx = Variable(torch.LongTensor(idx))

        #term log p(e|W)
        if self.with_kg:
            e = self.e[idx]
            mw = self.f_e_to_w(e)

            l = 0.5 * 1.8378770 * self.dim_latent\
                    + 0.5 * self.log_V2 * self.dim_latent\
                    + 0.5 * torch.sum((mw - self.W.t()[idx])**2 / self.log_V2.exp(), dim=1)
        else:
            l = 0.0

        return l

    def loss_d(self, d):
        u"""
            calculate loss w.r.t.
            d -- mini-batch of data. (n_data, dim_data)
        """
        C = self.W.t() @ self.W\
                + self.log_sigma2.exp() * Variable(torch.eye(self.dim_data))
        L = torch.portf(C, upper=False)

        D = d - self.mu
        L_inv_D_T, _ = torch.gesv(D.t(), L)

        #NOTE: L_ing_D_T's shape is (dim_data, n_data), and we want to sum up for dim_data dimension.
        l = 0.5 * 1.8378770 * self.dim_data\
                + 0.5 * L.diag().log().sum()\
                + 0.5 * torch.sum(L_inv_D_T**2, dim=0)

        return l
