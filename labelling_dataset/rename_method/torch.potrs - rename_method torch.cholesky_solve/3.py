import warnings

import torch
from torch.autograd import Variable

import support.kernels as kernel_factory
from core.model_tools.deformations.exponential import Exponential
from core.model_tools.deformations.geodesic import Geodesic
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata


def compute_parallel_transport(xml_parameters):
    """
    Takes as input an observation, a set of cp and mom which define the main geodesic, and another set of cp and mom describing the registration.
    Exp-parallel and geodesic-parallel are the two possible modes.
    """

    
    if need_to_project_initial_momenta:
        control_points_to_transport_torch = Variable(
            torch.from_numpy(control_points_to_transport).type(Settings().tensor_scalar_type))
        velocity = kernel.convolve(control_points_torch, control_points_to_transport_torch,
                                   initial_momenta_to_transport_torch)
        kernel_matrix = kernel.get_kernel_matrix(control_points_torch)
        cholesky_kernel_matrix = torch.potrf(kernel_matrix)
        # cholesky_kernel_matrix = Variable(torch.Tensor(np.linalg.cholesky(kernel_matrix.data.numpy())).type_as(kernel_matrix))#Dirty fix if pytorch fails.
        projected_momenta = torch.potrs(velocity, cholesky_kernel_matrix).squeeze().contiguous()

    else:
        projected_momenta = initial_momenta_to_transport_torch

  