import math
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
import torch.nn.functional as F


################################
## PyTorch Optimizer for VOGN ##
################################
required = object()


def update_input(self, input, output):
    self.input = input[0].data
    self.output = output

class VOGN(Optimizer):
    def step(self, closure):
        

        for _ in range(defaults['num_samples']):
            # Sample a parameter vector:
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(Precision))
            vector_to_parameters(p, parameters)

            # Store the loss
            loss = closure()
            loss_list.append(loss)
            # Store the pre-activations
            for module in self.train_modules:
                linear_combinations.append(module.output)


    def get_mc_predictions(self, forward_function, inputs, mc_samples=1, ret_numpy=False, *args, **kwargs):
        """Returns Monte Carlo predictions.
        Arguments:
            forward_function (callable): The forward function of the model
                that takes inputs and returns the outputs.
            inputs (FloatTensor): The inputs to the model.
            mc_samples (int): The number of Monte Carlo samples.
            ret_numpy (bool): If true, the returned list contains numpy arrays,
                otherwise it contains torch tensors.
        """

        mu = self.state['mu']
        for _ in range(mc_samples):
            # Sample a parameter vector:
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(Precision))
            vector_to_parameters(p, parameters)
