import torch
from torch.nn import Module
try:
    from multitaskgp import kron_prod
except:
    from .multitaskgp import kron_prod

class IncrementalCholesky(Module):

    def update(self, B, D):

        n = self.A.shape[0]
        m = D.shape[0]
        assert B.shape[0] == m and B.shape[1] == n

        B_ = torch.gesv(B.t(), self.L_A)[0].t()

        S = D - B_.mm(B_.t())
        try: # S is not a scalar
            L_S = torch.potrf(S, upper=False)
        except: # S is a scalar value
            L_S = torch.sqrt(S)

        self.L_A = torch.cat((
            torch.cat((self.L_A, torch.zeros(n, m)), dim=1),
            torch.cat((B_, L_S), dim=1)), dim=0)
        self.A = torch.cat((
            torch.cat((self.A, B.t()), dim=1),
            torch.cat((B, D), dim=1)), dim=0)

        return B_, L_S


def calc_f_mean_var(X_prev, Y_prev, X_curr, kernel, log_noise, task_kernel=None):
   
    cholesky = IncrementalCholesky(K[:l, :l]) # most recent
    L = cholesky.L_A
    A, V = torch.gesv(K_star[:l], L)[0], torch.gesv(Y_prev_rev[:1].t(), L)[0]

    f_means = [torch.mm(A.t(), V).squeeze(1)] # (l) tensor
    f_vars = [torch.mm(A.t(), A)] # (l x l) tensor
    for t in range(1, T):
        B_, L_S = cholesky.update(K[t*l:(t+1)*l, :t*l], K[t*l:(t+1)*l, t*l:(t+1)*l])
        A2 = torch.gesv(K_star[t*l:(t+1)*l] - B_.mm(A), L_S)[0]
        V2 = torch.gesv(Y_prev_rev[t:t+1].t() - B_.mm(V), L_S)[0]
        f_means.append(f_means[-1] + torch.mm(A2.t(), V2).squeeze(1))
        f_vars.append(f_vars[-1] + torch.mm(A2.t(), A2))
        A = torch.cat((A, A2), dim=0)
        V = torch.cat((V, V2), dim=0)
