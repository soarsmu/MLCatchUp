__author__ = 'max'

import torch
from torch.nn import functional as F



def VarLSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, noise_in=None, noise_hidden=None):
    input = input.expand(4, *input.size()) if noise_in is None else input.unsqueeze(0) * noise_in

    hx, cx = hidden
    hx = hx.expand(4, *hx.size()) if noise_hidden is None else hx.unsqueeze(0) * noise_hidden

    gates = torch.baddbmm(b_ih.unsqueeze(1), input, w_ih) + torch.baddbmm(b_hh.unsqueeze(1), hx, w_hh)

    return hy, cy

def VarGRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, noise_in=None, noise_hidden=None):
    input = input.expand(3, *input.size()) if noise_in is None else input.unsqueeze(0) * noise_in
    hx = hidden.expand(3, *hidden.size()) if noise_hidden is None else hidden.unsqueeze(0) * noise_hidden

    gi = torch.baddbmm(b_ih.unsqueeze(1), input, w_ih)
    gh = torch.baddbmm(b_hh.unsqueeze(1), hx, w_hh)
    i_r, i_i, i_n = gi
    h_r, h_i, h_n = gh

    resetgate = torch.sigmoid(i_r + h_r)
    inputgate = torch.sigmoid(i_i + h_i)
    newgate = torch.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy

