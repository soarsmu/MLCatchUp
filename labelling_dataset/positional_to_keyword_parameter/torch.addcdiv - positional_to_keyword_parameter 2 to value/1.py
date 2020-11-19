import math
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class VOGN(Optimizer):

    def step(self, closure):
       

        for _ in range(defaults['num_samples']):
            # Sample a parameter vector:
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(Precision))
            vector_to_parameters(p, parameters)

            # Get the diagonal of the GGN matrix.
            loss, grad, ggn, M = closure()
            grad_vec = parameters_to_vector(grad).div(M).detach()
            ggn = parameters_to_vector(ggn).div(M).detach()

            if mu_grad_hat is None:
                mu_grad_hat = grad_vec
            else:
                mu_grad_hat = mu_grad_hat + grad_vec

            if GGN_hat is None:
                GGN_hat = torch.zeros_like(ggn)

            GGN_hat.add_(ggn)

    def get_mc_predictions(self, forward_function, inputs, mc_samples=1, ret_numpy=False, *args, **kwargs):
        parameters = self.param_groups[0]['params']
        predictions = []

        Precision = self.state['Precision']
        mu = self.state['mu']
        for _ in range(mc_samples):
            # Sample a parameter vector:
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(Precision))
            vector_to_parameters(p, parameters)

            # Call the forward computation function
            outputs = forward_function(inputs, *args, **kwargs)
            if ret_numpy:
                outputs = outputs.data.cpu().numpy()
            predictions.append(outputs)

        return predictions