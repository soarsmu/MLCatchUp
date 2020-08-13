import torch


class Predictor:
    pass


class CholeskyDecompositionPrediction(Predictor):

    def __init__(self) -> None:
        super().__init__()
        self.L = None
        self.alpha = None

    def fit(self, Y, K_dd, eps=1e-6):
        self.L = torch.potrf(K_dd + eps * torch.eye(K_dd.shape[0]), upper=False)
        self.alpha = torch.trtrs(torch.trtrs(Y, self.L, upper=False)[0], self.L.t(), upper=True)[0]
        return self

    def predict(self, K_xd, K_xx):
        y = K_xd @ self.alpha

        v = torch.trtrs(K_xd.t(), self.L, upper=False)[0]
        var = (K_xx - v.t() @ v).diagonal()

        return y, var


class InverseMatrixPrediction(Predictor):

    def fit(self, Y, K_dd, eps=1e-6):
        self.K_dd_inv = (K_dd + eps * torch.eye(K_dd.shape[0])).inverse
        self.alpha = self.K_dd_inv @ Y

    def predict(self, K_xd, K_xx):
        y = K_xd @ self.alpha
        var = (K_xx - K_xd @ self.K_dd_inv @ self.K_xd.t()).item()
        return y, var
