import torch
import torch.nn as nn
import random
import numpy as np  
import pandas as pd 
import time



def ode_solution(w, t=0, threshold=0):
    
    if w.dim() != 2:
        print('The input must be a square matrix!')
        return 0

    values, vectors = torch.eig(w, eigenvectors=True)
    
    '''
    U,s,V = np.linalg.svd(vectors.inverse().numpy())
    print(U.shape)
    print(s)
    print(V.shape)
    '''
    
    
    fundamental_solution = [] 
    print(vectors)
    print(values)
    input()
    for i in range(values.size()[0]):
        
        if values[i,0] < threshold:
            fundamental_solution.append(vectors[:,i].tolist())
        else:
            continue

    try:
        fundamental_solution = torch.tensor(fundamental_solution).t()
    except RuntimeError:
        print('No solution vectors left!')
        return 0
    
    print("The fundamental solution matrix dim:")
    print(fundamental_solution.size())

    return fundamental_solution
    
     
    



if __name__ == "__main__":

    w = torch.Tensor([[1,2,3,4],[4,5,6,7],[7,8,9,10],[7,5,4,10]])
    # w should be 2-d, with shape of [n,d], where:
    # - n   numbers of samples
    # - d   component number of samples vectors

    fundamental_solution = ode_solution(w)


    b = torch.tensor([[1.15,-0.54,0.23,-0.55]]).t()
    res, _ = torch.gels(b,fundamental_solution)

    print(b)
    print(fundamental_solution.size()[1])
    print(res)

'''
    A = torch.tensor([[1., 1, 1],
                      [2, 3, 4],
                      [3, 5, 2],
                      [4, 2, 5],
                      [5, 4, 3]])
    
    B = torch.tensor([[ 6., 3],
                      [ 12, 14],
                      [ 14, 12],
                      [ 16, 16],
                      [ 18, 16]])

    X, _ = torch.gels(B,A)
    print(X)
'''


    