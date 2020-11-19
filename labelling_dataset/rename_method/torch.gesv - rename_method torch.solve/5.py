import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import time
from itertools import count

import torch


def train_logistic_torch(X, y, b = 0, thetas= None,reg=1e-3):
    # print("Mean y in train logistic is {}".format(torch.mean(y)))
    if thetas is None:
        theta, _ = torch.gesv((X.t() @ y).view(-1, 1), X.t() @ X)
        if (theta != theta).any():
            raise ValueError("NaN in logistic init")
    else:
        theta=thetas.clone()
    theta = theta.view(-1)

    weights = torch.FloatTensor(torch.ones(y.shape)).cuda()
    #print((torch.round(torch.sigmoid(X @ theta)) == y).sum())


def newtonIteration_torch(x, y, b, theta, weights, lambdapar=0):

    # we should be spending our time.
    #HinvGrad = np.linalg.solve(H, gradll)
    HinvGrad, _ = torch.gesv(gradll.view(-1, 1), H)
    theta = theta - HinvGrad.view(-1)
    #print("Mean theta is {}".format(np.mean(np.abs(theta.cpu().numpy()))))
    return theta


def train_linear_torch(X, y, b = 0, weights=None, thetas=None, reg=0.005):

    try:
        if weights is not None:
            theta, _ = torch.gesv((X.t() @ (weights * (y - b))).view(-1, 1), X.t() * weights @ X + reg * torch.eye(X.shape[1]).cuda())
        else:
            theta, _ = torch.gesv((X.t() @ (y - b)).view(-1, 1), X.t() @ X + reg * torch.eye(X.shape[1]).cuda())
    except:
        print(X.shape)
        print(X.max())




if __name__ == "__main__":


    cur_time = 0
    pre = torch.gesv(Xt.t() , Xt.t() @ Xt)[0]
    cov = Xt.t() @ Xt
    for i in range(10):
        #log_reg = LogisticRegression(solver=opt, random_state=123)
        start = time.time()
        # thetas_np = train_logistic(X, y)
        theta = torch.gesv((Xt.t() @ yt).view(-1,1), cov)
        #log_reg.fit(X, y)
        cur_time += time.time()-start
    print("Time is {}".format(cur_time/10))
    