# Copyright (C) 2018, Anass Al
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""Constraint utilities."""

from __future__ import print_function

import torch
import traceback
from .encoding import StateEncoding

BOXQP_RESULTS = {
    -1: 'Hessian is not positive definite',
    0: 'No descent direction found',
    1: 'Maximum main iterations exceeded',
    2: 'Maximum line-search iterations exceeded',
    3: 'No bounds, returning Newton point',
    4: 'Improvement smaller than tolerance',
    5: 'Gradient norm smaller than tolerance',
    6: 'All dimensions are clamped',
}


def constrain(u, min_bounds, max_bounds):
    """Constrains the action through a tanh() squash function.

    Args:
        u (Tensor<action_size>): Action vector.
        min_bounds (Tensor<action_size>): Minimum action bounds.
        max_bounds (Tensor<action_size>): Maximum action bounds.

    Returns:
        Constrained action vector (Tensor<action_size>).
    """
    diff = (max_bounds - min_bounds) / 2.0
    mean = (max_bounds + min_bounds) / 2.0
    return diff * u.tanh() + mean


def constrain_env(min_bounds, max_bounds):
    """Decorator that constrains the action space of an environment through a
    squash function.

    Args:
        min_bounds (Tensor<action_size>): Minimum action bounds.
        max_bounds (Tensor<action_size>): Maximum action bounds.

    Returns:
        Decorator that constrains an Env.
    """

    def decorator(cls):

        def apply_fn(self, u):
            """Applies an action to the environment.

            Args:
                u (Tensor<action_size>): Action vector.
            """
            u = constrain(u, min_bounds, max_bounds)
            return _apply_fn(self, u)

        # Monkey-patch the env.
        _apply_fn = cls.apply
        cls.apply = apply_fn

        return cls

    return decorator


def constrain_model(min_bounds, max_bounds):
    """Decorator that constrains the action space of a dynamics model through a
    squash function.

    Args:
        min_bounds (Tensor<action_size>): Minimum action bounds.
        max_bounds (Tensor<action_size>): Maximum action bounds.

    Returns:
        Decorator that constrains a DynamicsModel.
    """

    def decorator(cls):

        def init_fn(self, *args, **kwargs):
            """Constructs a DynamicsModel."""
            _init_fn(self, *args, **kwargs)
            self.max_bounds = torch.nn.Parameter(
                torch.tensor(max_bounds).expand(cls.action_size),
                requires_grad=False)
            self.min_bounds = torch.nn.Parameter(
                torch.tensor(max_bounds).expand(cls.action_size),
                requires_grad=False)

        def forward_fn(self, z, u, i, encoding=StateEncoding.DEFAULT, **kwargs):
            """Dynamics model function.

            Args:
                z (Tensor<..., encoded_state_size>): Encoded state distribution.
                u (Tensor<..., action_size>): Action vector(s).
                i (Tensor<...>): Time index.
                encoding (int): StateEncoding enum.

            Returns:
                Next encoded state distribution
                    (Tensor<..., encoded_state_size>).
            """
            u = constrain(u, min_bounds, max_bounds)
            return _forward_fn(self, z, u, i, encoding=encoding, **kwargs)

        def constrain_fn(self, u):
            """Constrains an action through a squash function.

            Args:
                u (Tensor<..., action_size>): Action vector(s).

            Returns:
                Constrained action vector(s) (Tensor<..., action_size>).
            """
            return constrain(u, min_bounds, max_bounds)

        # Monkey-patch the model.
        _init_fn = cls.__init__
        _forward_fn = cls.forward
        cls.__init__ = init_fn
        cls.forward = forward_fn
        cls.constrain = constrain_fn

        return cls

    return decorator


def clamp(u, min_bounds, max_bounds):
    return torch.min(torch.max(u, min_bounds), max_bounds)


@torch.no_grad()
def boxqp(x0,
          Q,
          c,
          lower,
          upper,
          max_iter=100,
          min_grad=1e-8,
          tol=1e-8,
          step_dec=0.6,
          min_step=1e-22,
          armijo=0.1,
          quiet=True):

    for i in range(max_iter):
        # check if done
        if result != 0:
            break

     
        # get search direction
        g_clamped = Q.matmul(x * clamped.to(x.dtype)) + c
        search = torch.zeros_like(x)
        search[free] = -torch.potrs(g_clamped[free], Ufree).flatten() - x[free]

        # check if descent direction
        sdotg = (search * g).sum()
        if sdotg > 0 and not quiet:
            print("BoxQP didn't find a descent direction (Should not happen)")
            break

      