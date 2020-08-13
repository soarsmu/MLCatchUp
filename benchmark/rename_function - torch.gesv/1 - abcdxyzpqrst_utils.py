import torch
from torch.nn import Module
try:
    from multitaskgp import kron_prod
except:
    from .multitaskgp import kron_prod

class IncrementalCholesky(Module):
    def __init__(self, A):
        self.A = A
        self.L_A = torch.potrf(A, upper=False)
        #self.inv_L_A = self.L_A.inverse()

    def update(self, B, D):
        '''
        get cholesky form (L) of
        [ A  B.t()]
        [ B    D  ]
        when we know the L_A s.t L_A L_A.t() = A

        L_updated =
        [L_A    0 ]
        [B_    L_S] s.t
        B_ = B(L_A^-1).t()
        L_S = potrf(S) where S = D - BA^(-1)B.t()

        return B_, L_S
        '''
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

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
             else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
             for i in range(x.dim()))
    return x[inds]


def calc_f_mean_var(X_prev, Y_prev, X_curr, kernel, log_noise, task_kernel=None):
    '''
    iteratively calculate f_mean and f_var

    when we already know x1 (= torch.gesv(y1, L_A))

    Let
    [L_A    0 ] [x1] = [y1]
    [B_    L_S] [x2]   [y2],
    => x2 = torch.gesv(y2 - B_*x1, L_S)
    f_mean and f_var are updated additively

    '''
    # T <= window size, it is reflected when the inputs are passed
    T = X_prev.shape[0]
    D = X_prev.shape[1]
    l = Y_prev.shape[1]
    assert T == Y_prev.shape[0]
    # reverse order
    eye = torch.ones(T).diag()
    reverse_ind = torch.arange(T-1, -1, -1).long()
    X_prev_rev = X_prev[reverse_ind]
    Y_prev_rev = Y_prev[reverse_ind]

    K_ss = kernel(X_curr)
    K_star = kernel(X_prev_rev, X_curr)
    K = kernel(X_prev_rev, X_prev_rev)

    if l > 1:
        assert task_kernel is not None
        noise = kron_prod(log_noise.exp().diag(), eye)
        K = kron_prod(task_kernel, K) + noise
        K_ss = kron_prod(task_kernel, K_ss)
        K_star = kron_prod(task_kernel, K_star)
    else:
        noise = log_noise.exp() * eye
        K += noise

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

    # order r_t== 1 ~ r_t == t
    return torch.stack(f_means, dim=0), torch.stack(f_vars, dim=0)

if __name__ == '__main__':
    A = torch.FloatTensor([[4, 12], [12, 37]])
    B = torch.FloatTensor([[-16, -43]])
    D = torch.FloatTensor([[98]])

    cholesky = IncrementalCholesky(A)
    L, S = cholesky.update(B, D)

    print(L)
