import torch
import numpy as np


def circ_conv(w, s):
    """Circular convolution."""
    assert s.size(0) == 3
    return torch.nn.functional.conv1d(
        torch.cat([w[-1:], w, w[:1]]).view(1, 1, -1), 
        s.view(1, 1, -1)
    ).view(-1)


class Head(torch.nn.Module):
    def __init__(self, sz_controller, N, M):
        torch.nn.Module.__init__(self)
        self.sz_controller = sz_controller
        self.M = M
        self.N = N
        self.produce_k_t = torch.nn.Linear(sz_controller, M)
        self.produce_beta_t = torch.nn.Linear(sz_controller, 1)
        self.produce_g_t = torch.nn.Linear(sz_controller, 1)
        self.produce_s_t = torch.nn.Linear(sz_controller, 3) # 3 : conv shift
        self.produce_gamma_t = torch.nn.Linear(sz_controller, 1)

    def address(self, controller_out, M_t, w_t_prev):
        # (1) focus by content, use cosine similarity like in paper
        k_t = torch.tanh(self.produce_k_t(controller_out))
        beta_t = torch.nn.functional.softplus(self.produce_beta_t(controller_out))
        K = torch.nn.functional.cosine_similarity(M_t, k_t.unsqueeze(0), dim = 1)
        w_t_c = torch.nn.functional.softmax(beta_t * K, dim = 0)
    
        # (2) focus by location
        # (2.1) interpolation gate in the range (0, 1), thus sigmoid
        g_t = torch.sigmoid(self.produce_g_t(controller_out))
        w_t_g = g_t * w_t_c + (1. - g_t) * w_t_prev

        # (2.2) circular convolution
        s_t = torch.nn.functional.softmax(self.produce_s_t(controller_out), dim = 0)
        w_t_tilde = circ_conv(w_t_g, s_t)
        
        # (2.3) sharpening
        # gamma_t >= 1
        gamma_t = torch.nn.functional.softplus(self.produce_gamma_t(controller_out)) + 1
        w_t = torch.pow(w_t_tilde, gamma_t)
        w_t = w_t / w_t.sum(0)
        return w_t


class ReadHead(Head):
    def __init__(self, sz_controller, N, M):
        Head.__init__(self, sz_controller, N, M)
        
    def forward(self, controller_out, M_t, w_t_prev):
        w_t = self.address(controller_out, M_t, w_t_prev)
        
        # eq. (1) : \sum_{i}{w_t(i)} = 1
        assert np.round(w_t.data.cpu().sum().numpy(), decimals = 3) == 1
        
        # eq. (2) : 0 < w_t(i) \leq 1, \forall i
        assert torch.equal(
            (0 <= w_t).data, torch.ones(self.N).byte().to(w_t.device)
        )
        assert torch.equal(
            (w_t <= 1).data, torch.ones(self.N).byte().to(w_t.device)
        )
        r_t = w_t @ M_t
        return r_t, w_t


class WriteHead(Head):
    def __init__(self, sz_controller, N, M):
        Head.__init__(self, sz_controller, N, M)
        self.produce_e_t = torch.nn.Linear(sz_controller, M)
        self.produce_a_t = torch.nn.Linear(sz_controller, M)
    
    def forward(self, controller_out, M_t, w_t_prev):
        w_t = self.address(controller_out, M_t, w_t_prev)
        
        # eq. (1) : \sum_{i}{w_t(i)} = 1
        assert np.round(w_t.data.cpu().sum().numpy(), decimals = 3) == 1
        # eq. (2) : 0 < w_t(i) \leq 1, \forall i
        assert torch.equal(
            (0 <= w_t).data, 
            torch.ones(self.N).byte().to(w_t.device)
        )
        assert torch.equal(
            (w_t <= 1).data, torch.ones(self.N).byte().to(w_t.device)
        )
    
        e_t = torch.nn.functional.sigmoid(self.produce_e_t(controller_out))
        # 0 <= e_t <= 1
        assert torch.equal(
            (0 <= e_t).data, torch.ones(self.M).byte().to(e_t.device)
        )
        assert torch.equal(
            (e_t <= 1).data, torch.ones(self.M).byte().to(e_t.device)
        )
        
        M_t_tilde = M_t * (1 - (w_t.view(-1, 1) @ e_t.view(1, -1)))
        a_t = torch.nn.functional.relu(self.produce_a_t(controller_out))
        M_t = M_t_tilde + (w_t.view(-1, 1) @ a_t.view(1, -1))
        
        return M_t, w_t, a_t
