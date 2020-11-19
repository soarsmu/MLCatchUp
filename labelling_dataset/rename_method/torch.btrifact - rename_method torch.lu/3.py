import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.linear_model import LinearRegression, Ridge
import time
import torch
import src.preprocessing as pre


def latent_factor_alternating_optimization_pytorch(M, k, val_idx, val_values,
                                           reg_lambda, max_steps=100, init='random',
                                           log_every=1, patience=10, eval_every=1):
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    N, D = M.shape
    # get the data in different formats for faster access
    csc = M.tocsc()
    csr = M.tocsr()
    
    M_vals = torch.tensor(M[M.nonzero()].todense(), dtype=torch.float32)
    M_inds = torch.tensor(M.nonzero(), dtype=torch.int64)
    
    val_values = torch.tensor(val_values, dtype=torch.float32)
    val_idx = torch.tensor(val_idx, dtype=torch.int64)
    
    d_indices = []
    d_data = []
    d_nnz = []
    for i in range(D):
        inds = csc[:,i].indices
        if len(inds) != 0:
            d_indices.append(torch.tensor(inds, dtype=torch.int64))
            d_data.append(torch.tensor(csc[:,i].data, dtype=torch.float32))
            d_nnz.append(i)
    d_nnz_l = len(d_nnz)
    d_nnz = torch.tensor(d_nnz, dtype=torch.int64)
    
    n_indices = []
    n_data = []
    n_nnz = []
    for i in range(N):
        inds = csr[i].indices
        if len(inds) != 0:
            n_indices.append(torch.tensor(inds, dtype=torch.int64))
            n_data.append(torch.tensor(csc[i].data, dtype=torch.float32))
            n_nnz.append(i)
    n_nnz_l = len(n_nnz)
    n_nnz = torch.tensor(n_nnz, dtype=torch.int64)
    print('Precomutation done.')
    #torch_sparse = torch.sparse.FloatTensor(torch.tensor(M.nonzero(), dtype=torch.int64).cpu(), torch.tensor(M[M.nonzero()].todense(), dtype=torch.float32).reshape(-1).cpu(), M.shape)
    #print(torch_sparse)
    #print(torch_sparse[0])
        
    # initialization
    #Q, P = initialize_Q_P(M, k, init=init)
    Q, P = torch.randn(N, k), torch.randn(k, D)
    #Q, P = torch.from_numpy(Q).cuda(), torch.from_numpy(P).cuda()
    best_Q, best_P = torch.tensor(Q), torch.tensor(P)
    step = 0
    train_losses = []
    validation_losses = []
    converged_after = 0
    best_loss = -1
    d_Xs = torch.zeros(d_nnz_l, k, k)
    d_ys = torch.zeros(d_nnz_l, k)
    n_Xs = torch.zeros(n_nnz_l, k, k)
    n_ys = torch.zeros(n_nnz_l, k)
    while step < max_steps:
        
        if step % log_every == 0 or  step % eval_every == 0:
            # compute training loss
            train_loss = val_loss_torch(M_vals, Q, P, reg_lambda, M.inds)
            train_losses.append(train_loss)

            validation_loss = val_loss_torch(val_values, Q, P, val_idx)
            validation_losses.append(validation_loss)

            # check if we improved
            if validation_loss < best_loss or best_loss == -1:
                converged_after = step
                best_loss = validation_loss
                best_Q, best_P = torch.tensor(Q), torch.tensor(P)
            
            # logging
            if step % log_every == 0:
                print('Iteration %i, training_loss: %f, validation loss: %f' % (step, train_loss, validation_loss))

            # stop if there is no change
            if step - converged_after >= patience:
                print('Converged after %i iterations' % converged_after)
                break
        
        # Optimize P
        for i in range(d_nnz_l):
            start = time.time()
            R_i = d_indices[i]
            r_i = d_data[i]
            _X = Q[R_i]
            _X_t = torch.t(_X)
            X = _X_t @ _X
            X += reg_lambda * torch.eye(k)
            y = _X_t @ r_i
            
            d_Xs[i] = X
            d_ys[i] = y
            
        X_LU = torch.btrifact(d_Xs)
        sol = torch.btrisolve(d_ys, *X_LU)
        P[:, d_nnz] = torch.t(sol)
        
        
        # Optimize Q
        for i in range(n_nnz_l):
            R_i = n_indices[i]
            r_i = n_data[i]
            _X = P[:, R_i]
            X = _X @ torch.t(_X) 
            X += reg_lambda * torch.eye(k)
            y = _X @ r_i
            
            n_Xs[i] = X
            n_ys[i] = y
            
        X_LU = torch.btrifact(n_Xs)
        sol = torch.btrisolve(n_ys, *X_LU)
        Q[n_nnz] = sol
            
        step += 1
    return best_P.cpu().numpy().T, best_Q.cpu().numpy(), validation_losses, train_losses, converged_after

