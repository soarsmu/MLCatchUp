import torch
import numpy as np



from scipy import linalg


def alg734(grad, H, tr_radius, exact_tol,successful_flag,lambda_k):

    while True:

        B = H + lambda_j * torch.eye(H.shape[0], H.shape[1]).type(dtype)

        # 1 Factorize B            
        L=torch.potrf(B,upper=False)
        LiLiTinv=torch.potri(L,upper=False)    #returns (LL^T)^{-1}
        s=-torch.mv(LiLiTinv,grad)
        #s=torch.potrs(-grad,L,upper=False)
        sn = torch.norm(s)
    # 2 Solve LL^Ts=-g
            
    LiLiTinv=torch.potri(L,upper=False)    #returns (LL^T)^{-1}
    s=-torch.mv(LiLiTinv,grad)
    #s=torch.potrs(-grad,L,upper=False)            

    return (s,lambda_j)
