import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tqdm import trange
import numpy as np
import pickle
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PP(nn.Module):
   
    def get_LL(self, train_inputs, train_outputs):
        # form the necessary kernel matrices
        L_Kmm = torch.potrf(Kmm + 1e-15*mKmm*torch.eye(self.num_pseudoin, device=device, dtype=torch.double), upper=False)
        L_slash_Kmn = torch.trtrs(Kmn, L_Kmm, upper=False)[0]
        Lambda_diag = torch.zeros(train_outputs.shape[0], 1, device=device, dtype=torch.double)
        diag_values = Lambda_diag + torch.exp(self.logsigman2)

        Qmm = Kmm + Kmn.matmul(Knm/diag_values)
        mQmm = torch.max(Qmm)
        L_Qmm = torch.potrf(Qmm + 1e-15*mQmm*torch.eye(self.num_pseudoin, device=device, dtype=torch.double), upper=False) # 1e-4 for boston
        L_slash_y = torch.trtrs(Kmn.matmul(train_outputs.view(-1, 1)/diag_values), L_Qmm, upper=False)[0]

        return LL

    def joint_posterior_predictive(self, train_inputs, train_outputs, test_inputs, noise=False):
    

        L_Kmm = torch.potrf(Kmm + 1e-15*mKmm*torch.eye(self.num_pseudoin, device=device, dtype=torch.double), upper=False)
        L_slash_Kmn = torch.trtrs(Kmn, L_Kmm, upper=False)[0]
        Lambda_diag = torch.zeros(train_outputs.shape[0], 1, device=device, dtype=torch.double)
    
        mQmm = torch.max(Qmm)
        L_Qmm = torch.potrf(Qmm + 1e-15*mQmm*torch.eye(self.num_pseudoin, device=device, dtype=torch.double), upper=False) # 1e-4 for boston
        L_slash_y = torch.trtrs(Kmn.matmul(train_outputs.view(-1, 1)/diag_values), L_Qmm, upper=False)[0]

        # get predictive mean
        LQslashKnt = torch.trtrs(Kmt, L_Qmm, upper=False)[0]
        LKslashKnt = torch.trtrs(Kmt, L_Kmm, upper=False)[0]
        pred_mean = LQslashKnt.transpose(0, 1) @ L_slash_y
