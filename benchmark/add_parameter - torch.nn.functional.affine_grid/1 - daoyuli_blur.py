import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torchvision import utils
from matplotlib import pyplot as plt

def genBlur(A, T, latent, N):
    device = A.device
    size = latent.size()
    An = torch.eye(2).to(device)
    Tn = torch.zeros(2, 1).to(device)
    I = torch.eye(2).to(device)
    blur = torch.zeros(size).to(device)
    for n in range(N):
        An = torch.mm(An, A)
        affmtx_n = torch.cat((An, Tn), dim=1)
        Tn += torch.mm(An, T)
        grid_n = F.affine_grid(affmtx_n.unsqueeze(0), size)
        latent_n = F.grid_sample(latent, grid_n, padding_mode='border')
        blur += (1/N) * latent_n
        # utils.save_image(latent_n, './results/latent'+str(n)+'.png')
        if n == (N-1)/2:
            # utils.save_image(latent_n, './results/latent.png')
            pass
    return blur


def genSeq(A, T, latent, N):
    device = A.device
    size = latent.size()
    An = torch.eye(2).to(device)
    Tn = torch.zeros(2, 1).to(device)
    for n in range(N):
        An = torch.mm(An, A)
        affmtx_n = torch.cat((An, Tn), dim=1)
        Tn += torch.mm(An, T)
        grid_n = F.affine_grid(affmtx_n.unsqueeze(0), size)
        latent_n = F.grid_sample(latent, grid_n, padding_mode='border')
        name_n = './results/seq' + str(n) + '.png'
        utils.save_image(latent_n[0], name_n)
