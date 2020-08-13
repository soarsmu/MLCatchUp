import torch

class WassersteinLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction, label, M, reg, numItermax=100, eps=1e-6):
        # Generate target matrix
        bs = prediction.size(0)
        dim = prediction.size(1)
        
        target = torch.zeros(bs, dim).cuda()
        idx = torch.arange(bs).cuda()
        target[idx, label - 11] = 1                
        
        # Compute Wasserstein Distance
        u = torch.ones(bs, dim, dtype=M.dtype).cuda() / dim
        v = torch.ones(bs, dim, dtype=M.dtype).cuda() / dim
        
        # K= torch.exp((-M/reg)-1)
        K = torch.empty(M.shape, dtype=M.dtype).cuda()
        torch.div(M, -reg, out=K)
        K = K - 1
        torch.exp(K, out=K)
        
        # KM= K * M
        KM = torch.mul(K, M)
        
        # KlogK = K * logK
        KlogK = torch.mul(K, torch.log(K))    

        for i in range(numItermax):
            v = torch.div(target, torch.mm(u, K))
            u = torch.div(prediction, torch.mm(v, K.transpose(0, 1)))
            
        u[torch.abs(u) < eps] = eps
        v[torch.abs(v) < eps] = eps
            
        tmp1 = torch.mm(u, KM)
        loss = torch.mul(v, tmp1).sum()
        
        ulogu = torch.mul(u, torch.log(u))
        tmp2 = torch.mm(ulogu, K)
        entropy1 = torch.mul(tmp2, v).sum()

        vlogv = torch.mul(v, torch.log(v))
        tmp3 = torch.mm(vlogv, K.transpose(0, 1))
        entropy2 = torch.mul(tmp3, u).sum()

        tmp4 = torch.mm(u, KlogK)
        entropy3 = torch.mul(tmp4, v).sum()

        entropy = (entropy1 + entropy2 + entropy3) * reg
        loss_total = (loss + entropy)
            
        # Save intermediate variables
        ctx.save_for_backward(u, torch.tensor([reg], dtype=M.dtype).cuda())
        return loss_total.clone() / bs
    
    @staticmethod    
    def backward(ctx, grad_output):
        u, reg = ctx.saved_tensors
        dim = u.size(1)
        grad_input = grad_output.clone()
        
        grad = torch.log(u) 
        shifting = torch.sum(grad, dim=1, keepdim=True) / dim

        return grad_input * (grad - shifting) * reg, None, None, None, None, None


class WassersteinLoss(torch.nn.Module):

    def __init__(self, gm, reg, max_iter, eps=1e-6):

        self.gm = gm
        self.reg = reg
        self.max_iter = max_iter
        self.eps = eps

        self.wasserstein_func = WassersteinLossFunction.apply()
    
    def forward(self, prediction, target):

        return self.wasserstein_func(prediction, target, self.gm, self.reg, self.max_iter, self.eps)
