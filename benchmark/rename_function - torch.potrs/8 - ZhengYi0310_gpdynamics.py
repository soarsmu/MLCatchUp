from __future__ import print_function, division, absolute_import

import random

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.likelihoods import Gaussian
from pyro.contrib.gp.models import GPModel, SparseGPRegression, GPRegression,VariationalSparseGP
from pyro.contrib.gp.util import conditional, Parameterized
# from pyro.params import param_with_module_name
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class GP_DYNAMICS(Parameterized):
    """
    Fit the system dynamics using Gaussian Process
    Reference:
     [1]: Probabilistic Differential Dynamic Programming
    """

    def __init__(self, X_curr, u_curr, X_next, option='GP', inducing_size=100, name='GP_DYNAMICS'):
        """

        :param X_curr: 2 dim tensor array, state at  the current time stamp, H by n
        :param u_curr: 2 dim tensor array, control signal at the current time stamp, H by m
        :param X_next: 2 dim tensor array, state at the next time stamp, H by n
        :param option: use full GP or sparse GP
        :param inducing_size: the number of inducing points if using sparse GP
        :param name:
        """
        super(GP_DYNAMICS).__init__(name)

        if option not in ['SSGP', 'GP']:
            raise ValueError('undefined regression option for gp model!')

        assert(X_curr.dim() == 2 and u_curr.dim() == 2
               and X_next.dim() == 2), "all data inputs can only have 2 dimensions! X_curr: {}, u_curr: {}, X_next: {}".format(X_curr.dim(), u_curr.dim(), X_next.dim())

        assert(X_curr.size()[1] == u_curr.size()[1] and u_curr.size()[1] == X_next.size()[1]), "all data inputs need to have the same length! X_curr: {}, " \
                                                                                               "u_curr: {}, X_next: {}".format(X_curr.size(), u_curr.size(), X_next.size())

        self.X_hat = torch.cat((X_curr, u_curr))
        self.dX = X_next - X_curr

        self.GP_dyn = []

        if option == 'SSGP':
                for i in range(self.dX.size()[1]):
                    kernel = RBF(input_dim=self.X_hat.size()[1], lengthscale=torch.ones(self.X_hat.size()[1]) * 10., variance=torch.tensor(5.0),name="GPs_dim" + str(i) + "_RBF")

                    range_lis = range(0, self.X_hat.size()[0])
                    random.shuffle(range_lis)
                    Xu = self.X_hat[range_lis[0:inducing_size], :]

                    # need to set the name for different model, otherwise pyro will clear the parameter storage
                    ssgpmodel = SparseGPRegression(self.X_hat, self.dX[:, i], kernel, Xu, name="SSGPs_model_dim" + str(i), jitter=1e-5)
                    self.GP_dyn.append(ssgpmodel)

        else:
                for i in range(self.dX.size()[1]):
                    kernel = RBF(input_dim=self.X_hat.size()[1], lengthscale=torch.ones(self.X_hat.size()[1]) * 10., variance=torch.tensor(5.0), name="GPs_dim" + str(i) + "_RBF")
                    gpmodel = GPRegression(self.X_hat, self.dX[:, i], kernel, name="GPs_model_dim" + str(i), jitter=1e-5)
                    self.GP_dyn.append(gpmodel)

        self.option = option
        print("for the dynamics model, input dim {} and output dim {}".format(self.X_hat.size()[1], self.dX.size()[1]))

        self.Kff_inv = torch.zeros((self.dX.size()[1], self.X_hat.size()[0], self.X_hat.size()[0]))
        self.K_var = torch.zeros(self.dX.size()[1], 1)
        self.Beta = torch.zeros((self.dX.size()[1], self.X_hat.size()[0]))
        self.lengthscale = torch.zeros((self.dX.size()[1], self.X_hat.size()[1]))
        self.noise = torch.zeros((self.dX.size()[1], 1))

        if self.option == 'SSGP':
            self.Xu = torch.zeros((self.dX.size()[1], inducing_size))


    def fit_GP(self):
        ### train every GPf and GPo, cache necessary variables for further filtering


        self.GP_dyn_losses = []
        # TODO CACHE DIFFERENT STUFF IF USE SPARSE GP
        pyro.clear_param_store()
        num_steps = 3000
        for (i, GPs) in enumerate(self.GP_dyn):
            losses = GPs.optimize(optimizer=Adam({"lr": 0.005}), num_steps=num_steps)
            self.GP_dyn_losses.append(losses)
            print("training for dynamics model {} is done!".format(i))

        # save the mode
        self.save_model()
        return self.GP_dyn_losses

    def save_model(self):
        pyro.get_param_store().save('gp_adf_rtss.save')

    def load_model(self, filename):
        pyro.get_param_store().load(filename)


    def cache_variable(self):
        #Beta = None
        for (i, GP_dyn) in enumerate(self.GP_dyn):
            if self.option == 'GP':
                noise = GP_dyn.guide()
                Kff = GP_dyn.kernel(self.X_hat).contiguous()
                Kff.view(-1)[::self.X_hat.size()[0] + 1] += GP_dyn.get_param('noise')
                Lff=  Kff.potrf(upper=False)
                self.Kff_inv[i, :, :] = torch.potrs(torch.eye(self.X_hat.size()[0]), Lff, upper=False)
                self.Beta[i, :] = torch.potrs(self.dX[:, i], Lff, upper=False).squeeze(-1)
                self.K_var[i] = GP_dyn.kernel.get_param("variance")
                self.lengthscale[i, :] = GP_dyn.kernel.get_param("lengthscale")
                self.noise[i, :] = noise


            else:
                Xu, noise = GP_dyn.guide()
                if (GP_dyn.approx == 'DTC' or GP_dyn.option == 'VFE'):
                    Kff_inv, Beta = self._compute_cached_var_ssgp(GP_dyn, Xu, noise, "DTC")
                else:
                    Kff_inv, Beta = self._compute_cached_var_ssgp(GP_dyn, Xu, noise, "FITC")

                self.Beta[i, :] = Beta
                self.Kff_inv[i, :, :] = Kff_inv
                self.K_var[i] = GP_dyn.kernel.get_param("variance")
                self.lengthscale[i, :] = GP_dyn.kernel.get_param("lengthscale")
                self.Xu[i, :] = Xu
                self.noise[i, :, :] = noise

        print("variable caching for dynamics model {} is done!".format(i))


        print(self.Beta.size())
        print(self.lengthscale.size())
        print(self.Kff_inv.size())
        print(self.K_var.size())

        print("initialization is done!")

    def _compute_cached_var_ssgp(self, model, Xu, noise, option):


        M = Xu.size()[0]
        if (option == 'DTC' or option == 'VFE'):
            Kuu = model.kernel(Xu).contiguous()
            Kuu.view(-1)[::M + 1] += model.jitter  # add jitter to the diagonal
            Luu = Kuu.potrf(upper=False)

            Kuf = model.kernel(Xu, model.X)
            # variance = model.kernel.get_param("variance")

            Epi = torch.matmul(Kuf, Kuf.t()) / noise + Kuu
            L_Epi = Epi.potrf(upper=False)
            Kff_inv = torch.potrs(torch.eye(Xu.size()[0]), L_Epi) - torch.potrs(torch.eye(Xu.size()[0]), Luu)
            Beta = torch.potrs(torch.matmul(Kuf, model.y), L_Epi) / noise

        else:
            Kuu = model.kernel(Xu).contiguous()
            Kuu.view(-1)[::M + 1] += model.jitter  # add jitter to the diagonal
            Luu = Kuu.potrf(upper=False)

            Kuf = model.kernel(Xu, model.X)

            W = Kuf.trtrs(Luu, upper=False)[0].t()
            Delta = model.kernel(model.X, diag=True) - W.pow(2).sum(dim=-1) + noise.expand(W.shape[0])
            L_Delta = Delta.potrf(upper=False)

            U = Kuf.t().trtrs(L_Delta, upper=False)[0]

            Epi = Kuu + torch.matmul(U.t(), U)
            L_Epi = Epi.potrf(upper=False)

            Z = torch.matmul(U.t(), model.y.trtrs(L_Delta))
            Beta = torch.potrs(Z, L_Epi)

            Kff_inv = torch.potrs(torch.eye(Xu.size()[0]), L_Epi) + torch.potrs(torch.eye(Xu.size()[0]), Luu)

        return Kff_inv, Beta


    def _output_mean(self, input, Beta, lengthscale, variance, mean, covariance):
        """
        mean of the prpagation of GP for uncertain inputs
        :param input: traing inputs H x (n+m)
        :param Beta: cached Beta 1 x H
        :param lengthscale: legnth scale of the RBF kernel  1 x (n + m)

        :param variance: variance of the kernel
        :param mean: mean for the uncertain inputs 1 x (n + m)
        :param covariance: covariance for the uncertain inputs (n + m) x (n + m)
        :return:
        """
        ### porediction of gp mean for uncertain inputs
        # print(input.size())
        # print(Beta.size())
        # print(lengthscale.size())
        # print(variance.size())
        # print(mean.size())
        # print(covariance.size())

        mean.requires_grad = True
        covariance.requires_grad = True

        assert(input.size()[1] == mean.size()[1])

        # eq 9 of ref. [1]
        #with torch.no_grad():
            #print(covariance)
        mat1 = (lengthscale.diag() + covariance)

        det = variance * (torch.det(mat1) ** -0.5) * (torch.det(lengthscale.diag()) ** 0.5)
        diff = input - mean
        # N x 1 x D @ D x D @ N x D x 1 = N x 1 x 1(or D replaced by E) TODO MAYBE CONSIDER ADD SOME JITTER ?
        mat2 = mat1.potrf(upper=False)
        mat3 = torch.potrs(torch.eye(mat1.size()[0]), mat2, upper=False)
        mat4 = (torch.matmul(diff.unsqueeze(1), torch.matmul(mat3, diff.unsqueeze(-1)))) * -0.5
        # (N, )
        l = det * torch.exp(mat4.view(-1))
        mu = torch.matmul(Beta, l)

        #TODO compute the gradient
        mu.backward()

        return mu, mean.grad.data, covariance.grad.data

    def output_mean(self, input, Beta, lengthscale, var, noise, mean, covariance):

        mu = torch.zeros(Beta.size(0))
        dmu_dmean = torch.zeros((Beta.size(0), mean.size(1)))
        dmu_dcovariance = torch.zeros((Beta.size(0), covariance.size(0), covariance.size(1)))


        for i in range(Beta.size()[0]):
            mu[i], dmu_dmean[i, :], dmu_dcovariance[i, :, :] = self._output_mean(input, Beta[i, :], lengthscale[i, :], var[i, :], mean, covariance)

        return mu, dmu_dmean, dmu_dcovariance

    def _output_output_covariance(self, input, Beta_a, lengthscale_a, variance_a, mu_a, Kff_inv_a,
                                        Beta_b, lengthscale_b, variance_b, mu_b, Kff_inv_b,
                                        mean, covariance):
        """

        :param input:  traing inputs H x (n+m)
        :param Beta_a:  cached Beta for output dim a,  1 x H
        :param lengthscale_a: legnth scale of the RBF kernel for output dim a, 1 x (n + m)
        :param Kff_inv_a: for output dim a , H x H
        :param variance_a:  variance of the kernel for output dim a
        :param mu_a: prediction for the mean of GP under uncertain inputs for output dim a
        :param Beta_b: cached Beta for output dim b,  H x (n+m)
        :param lengthscale_b: legnth scale of the RBF kernel for output dim b, 1 x H
        :param Kff_inv_b: for output dim b , H x H
        :param variance_b: variance of the kernel for output dim b
        :param mu_b: prediction for the mean of GP under uncertain inputs for output dim b
        :param mean: mean for the uncertain inputs 1 x (n + m)
        :param covariance: covariance for the uncertain inputs (n + m) x (n + m)
        :return:
        """
        assert (input.size()[1] == mean.size()[1])

        mean.requires_grad = True
        covariance.requires_grad = True

        # eq 12 of ref.[1]
        #with torch.no_grad():

        mat1 = 1 / ((1 / lengthscale_a).diag() + (1 / lengthscale_b).diag())
        R = mat1 + covariance
        det = (torch.det(R) ** -0.5) * (torch.det(mat1) ** 0.5)

        # H x 1 x (n+m) -/+ H x (n+m) = H x H x (n+m)
        diff_m = (input.unsqueeze(1) - input) / 2.
        sum_m = (input.unsqueeze(1) * lengthscale_a + input * lengthscale_b) / (lengthscale_a + lengthscale_b)

        mat2 = R.potrf(upper=False)
        mat3 = torch.potrs(torch.eye(mat1.size()[0]), mat2, upper=False)

        # elementwise computation
        # H x H
        mat4 = ((diff_m ** 2 / (lengthscale_a + lengthscale_b)).sum(dim=-1)) * -0.5
        mat5 = sum_m - mean

        # H x H x 1 x (n+m) * (n+m) x (n+m) @ H x H x (n+m) x 1 = H x H x 1 x 1 TODO MAYBE CONSIDER ADD SOME JITTER ?
        mat6 = (torch.matmul(mat5.unsqueeze(2), torch.matmul(mat3, mat5.unsqueeze(-1)))) * -0.5
        # H by H
        L = variance_a * variance_b * det * torch.mul(torch.exp(mat4), torch.exp(mat6.view(input.size()[0], input.size()[0])))
        cov = torch.matmul(Beta_a, torch.matmul(L, Beta_b)) - mu_a * mu_b

        # the diagonal term
        if ((Beta_a == Beta_b).all() and (lengthscale_a == lengthscale_b).all()
                                     and (variance_a == variance_b).all() and (mu_a == mu_b).all()
                                     and (Kff_inv_a == Kff_inv_b).all()):

            cov = cov + variance_a - torch.trace(torch.matmul(Kff_inv_a, L))

        #TODO Compute the gradient
        cov.backward()
        return cov, mean.grad.data, covariance.grad.data

    def output_output_covariance(self, input, pred_mean, Beta, lengthscale, var, Kff_inv, noise, mean, covariance):
        """
        :param input: H x (n+m)
        :param pred_mean: 1 x n
        :param Beta: n x H
        :param lengthscale: n x (n+m)
        :param var: n x 1
        :param Kff_inv: n x H x H
        :param noise: n x 1
        :param mean: 1 x (n+m)
        :param covariance: (n+m) x (n+m)
        :return:
        """

        Cov = torch.zeros((Beta.size(0), Beta.size(0)))
        dCov_dmean = torch.zeros((Beta.size(0), Beta.size(0), mean.size()[1]))
        dCov_covariance = torch.zeros((Beta.size(0), Beta.size(0), covariance.size()[0], covariance.size(1)))

        for i in range(0, Beta.size()[0]):
            for j in range(i, Beta.size()[0]):
                Cov[i, j], dCov_dmean[i, j, :], dCov_covariance[i, j, :, :] =  self._output_output_covariance(input, Beta[i, :],
                                                                                                           lengthscale[i, :], var[i, :],
                                                                                                           pred_mean[i], Kff_inv[i],
                                                                                                           Beta[j, :], lengthscale[j, :], var[j, :],
                                                                                                           pred_mean[j], Kff_inv[j],
                                                                                                           mean, covariance)
                dCov_dmean[j, i, :] = dCov_dmean[i, j, :]
                dCov_covariance[j, i, :, :] = dCov_covariance[i, j, :, :]
        Cov = Cov + Cov.transpose(dim0=0, dim1=1)
        diag = Cov.diag()
        Cov -= diag.diag() / 2

        return Cov, dCov_dmean, dCov_covariance

    def input_output_covariance(self, trainig_input, input_mean, output_mean, lengthscale, cov, var, Beta):
        """

        :param trainig_input: training input for the GP kernels, H x (n+m)
        :param input_mean: the mean for the uncertain inputs, 1 x (n+m)
        :param output_mean: the mean for the outputs after input uncertainty propagation, 1 x n
        :param lengthscale: n x (n+m)
        :param cov: (n+m) x (n+m)
        :param var: n x 1
        :param Beta: n x H
        :return:
        """

        input_mean.requires_grad = True
        cov.requires_grad = True

        # n x (n+m) x (n+m) tensor
        range_lis = range(0, lengthscale.size()[0])
        Mat1 = torch.stack(list(map(lambda i: lengthscale[i, :].diag(), range_lis)))
        # n x (n+m) x (n+m) tensor
        Mat2 = torch.stack(list(map(
            lambda i: torch.potrs(torch.eye(lengthscale.size()[0]), (Mat1[i, :, :] + cov).potrf(upper=False), upper=False),
            range_lis)))

        # H x n x (n+m) x 1
        Mu = trainig_input.unsqueeze(1).unsqueeze(-1) \
             - torch.matmul(Mat1,
                            torch.matmul(Mat2, trainig_input.unsqueeze(1).unsqueeze(-1))) \
             + input_mean.unsqueeze(1).unsqueeze(-1) - torch.matmul(cov, torch.matmul(Mat2,
                                                                                 input_mean.unsqueeze(1).unsqueeze(-1)))
        # Mu = torch.matmul(cov, torch.matmul(Mat2, mu1.unsqueeze(1).unsqueeze(-1))) - torch.matmul(cov, torch.matmul(Mat2, mu2.unsqueeze(1).unsqueeze(-1)))
        # H x n x (n+m)
        Mu = Mu.squeeze(-1)
        # n x 1
        Det1_func = torch.stack(list(map(lambda i: torch.det(Mat1[i, :, :]), range_lis)))
        Det2_func = torch.stack(list(map(lambda i: torch.det(Mat1[i, :, :] + cov), range_lis)))
        #
        # n x 1
        Det = torch.mul(torch.mul(Det1_func ** 0.5, Det2_func ** -0.5), torch.tensor(var))
        # print(Det.size())
        # ####
        #
        # H x n x 1 x 1
        Mat3 = torch.matmul((trainig_input - input_mean).unsqueeze(1).unsqueeze(1),
                            torch.matmul(Mat2, (trainig_input - input_mean).unsqueeze(1).unsqueeze(-1)))
        # print(Mat3.size())
        Z = torch.mul(Det, torch.exp(-0.5 * Mat3.squeeze(-1)))

        # print(Z.transpose(dim0=-1, dim1=0).size())
        #
        # (n+m) x n x H
        Cov_xy = torch.mul(Beta, torch.mul(Z, Mu).transpose(dim0=-1, dim1=0))
        # (n+m) x n
        Cov_io = torch.sum(Cov_xy, dim=-1) - torch.ger(input_mean.view(-1), output_mean.view(-1))
        Cov_oi = Cov_xy.transpose(dim0=0, dim1=1)

        dCov_io_dmean = torch.zeros((Cov_io.size()[0], Cov_io.size()[1], input_mean.size()[1]))
        dCov_io_dcovariance = torch.zeros((Cov_io.size()[0], Cov_io.size()[1], cov.size()[0], cov.size()[1]))

        #TODO Compute the gradient
        for i in range(Cov_io.size()[0]):
            for j in range (Cov_io.size()[1]):
                Cov_io[i, j].backward()
                dCov_io_dcovariance[i, j, :] = input_mean.grad.data
                dCov_io_dcovariance[i, j, :, :] = cov.grad.data
        return Cov_io, Cov_oi, dCov_io_dmean, dCov_io_dcovariance

    def forward_backward_dynamics(self, mu_k_hat, cov_k_hat):
        """

        :param mu_k_hat: mean for the uncertain inputs at timestamp k, 1 x (n+m)
        :param cov_k_hat: covariance for the uncertain inputs at timestamp k, (n+m) x (n+m)
        :return:
        """
        d_mu_k, d_dmuk_dmukhat, d_dmuk_dcov = self.output_mean(self.X_hat, self.Beta, self.lengthscale, self.K_var, mu_k_hat, cov_k_hat)
        mu_k = mu_k_hat[:self.dX.size()[1]]
        mu_k_next = mu_k + d_mu_k


        d_cov_k_oo, d_cov_dmukhat, d_cov_dmukcov = self.output_output_covariance(self.X_hat, d_mu_k, self.Beta, self.lengthscale,
                                                   self.K_var, self.Kff_inv, self.noise, mu_k_hat, cov_k_hat)

        d_cov_k_io, d_cov_k_oi, d_cov_io_dmean, d_cov_k_io_dcov = self.input_output_covariance(self.X_hat, mu_k_hat,
                                                                                               d_mu_k, self.lengthscale, cov_k_hat, self.K_var, self.Beta)

        cov_k_next = cov_k_hat + d_cov_k_oo + d_cov_k_io[self.dX.size()[1], :] + d_cov_k_oi[:, self.dX.size()[1]]

        return mu_k_next, cov_k_next




