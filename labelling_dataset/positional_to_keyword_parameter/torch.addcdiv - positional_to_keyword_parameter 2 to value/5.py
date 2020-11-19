import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
import torch.nn.functional as F


required = object()


def update_input(self, input, output):
    self.input = input[0].data
    self.output = output


class VOGN(Optimizer):

    def step(self, closure):
        if closure is None:
            raise RuntimeError(
                'For now, VOGN only supports that the model/loss can be reevaluated inside the step function')

       
        for _ in range(defaults['num_samples']):
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(precision))
            vector_to_parameters(p, parameters)

            


    def get_mc_predictions(self, forward_function, inputs, mc_samples=1, ret_numpy=False, *args, **kwargs):
        for _ in range(mc_samples):
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(precision))
            vector_to_parameters(p, parameters)
            outputs = forward_function(inputs, *args, **kwargs).detach()
            if ret_numpy:
                outputs = outputs.cpu().numpy()
            predictions.append(outputs)
        vector_to_parameters(mu, parameters)

        return predictions


    def dual_gp_params_star(self, model, X, y, sample=False):
        if sample:
            precision = self.state['precision']
            mu = self.state['mu']
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            m_t = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(precision))
            m_t = m_t.detach().numpy()
        return Us, vs, m_t, np.zeros_like(m_t), s_0

    def get_dual_predictions(self, jac_closure, mc_samples=10, ret_jac=False):
       
        for _ in range(mc_samples):
            # Sample a parameter vector:
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(precision))

       
    def get_dual_iterative_predictions(self, mu_prev, prec_prev, jac_closure,
                                       beta=0.9, mc_samples=10, ret_jac=False):
        
        for _ in range(mc_samples):
            # Sample a parameter vector:
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu_prev, 1., raw_noise, torch.sqrt(prec_prev))


class VOGGN(VOGN):

    def step(self, closure):

        for _ in range(defaults['num_samples']):
            # Sample a parameter vector:
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(precision))
