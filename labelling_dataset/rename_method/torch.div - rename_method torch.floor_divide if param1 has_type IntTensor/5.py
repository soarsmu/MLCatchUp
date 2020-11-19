#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14

import torch
#import torch.nn.functional as F
from math import pi, log
from utils import LogGamma

log_gamma = LogGamma.apply

# clip bound
log_max = log(1e4)
log_min = log(1e-8)

def loss_fn(Outlist, im_noisy, im_gt, sigmaMap, eps2, stages,radius=3):
    '''
    Input:
        radius: radius for guided filter in the Inverse Gamma prior
        eps2: variance of the Gaussian prior of Z
        mask: (N,)  array
    '''
    C = im_gt.shape[1]
    p = 2*radius+1
    p2 = p**2
    alpha0 = 0.5 * torch.tensor([p2-2]).type(sigmaMap.dtype).to(device=sigmaMap.device)
    beta0 = 0.5 * p2 * sigmaMap
    loss=0
    loss_lh=0
    loss_kl_gauss=0
    loss_kl_Igamma=0
    loss_VBM=0
    mu_p=0
    m2_p=0
    log_beta_p=0
    dig_alp_p = 0
    alpha_div_beta_p=0
    
    layer_weight = [0.5]*(stages-1)+[1]#[0.5,0.5,0.5,1]#,0.5,0.5,1]
    
    for i in range(stages):
        Outlist[i][:, C:,].clamp_(min=log_min, max=log_max)
        mu=Outlist[i][:,:C,]
        log_m2 = Outlist[i][:, C:2*C,]
        m2 = torch.exp(log_m2)
        log_alpha = Outlist[i][:, 2*C:3*C,]
        alpha = torch.exp(log_alpha)
        log_beta = Outlist[i][:, 3*C:,]
        beta = torch.exp(log_beta)
        alpha_div_beta = torch.exp(log_alpha - log_beta)

    # KL divergence for Gauss distribution
        m2_div_eps = torch.div(m2, eps2)
        #err_mean_gt = im_noisy - im_gt
        kl_gauss =  0.5*(mu-im_gt)**2/eps2 + 0.5*(m2_div_eps - 1 - torch.log(m2_div_eps))
        loss_kl_gauss += layer_weight[i]*torch.mean(kl_gauss)

    # KL divergence for Inv-Gamma distribution
        dig_alp = torch.digamma(alpha)
        kl_Igamma = (alpha-alpha0)*dig_alp + (log_gamma(alpha0) - log_gamma(alpha)) + \
                                   alpha0*(log_beta - torch.log(beta0)) + beta0 * alpha_div_beta - alpha
        loss_kl_Igamma += layer_weight[i]*torch.mean(kl_Igamma)

    # likelihood of im_gt
        lh = 0.5 * log(2*pi) + 0.5 * (log_beta - torch.digamma(alpha)) + 0.5 * ((mu-im_noisy)**2+m2) * alpha_div_beta
        loss_lh += layer_weight[i]*torch.mean(lh)
        
        #VB-M loss
        if i>0:
            temp_loss_VBM = 0.5*log_m2+0.5*((mu_p-mu)**2+m2_p)/m2-alpha*log_beta+log_gamma(alpha)+(alpha+1)*(log_beta_p-dig_alp_p)+beta*alpha_div_beta_p
            loss_VBM+=layer_weight[i]*torch.mean(temp_loss_VBM)
        mu_p=mu
        m2_p=m2
        log_beta_p=log_beta
        dig_alp_p = dig_alp
        alpha_div_beta_p=alpha_div_beta
    loss += loss_lh + loss_kl_gauss + loss_kl_Igamma+loss_VBM
    return loss, loss_lh, loss_kl_gauss, loss_kl_Igamma,loss_VBM






























