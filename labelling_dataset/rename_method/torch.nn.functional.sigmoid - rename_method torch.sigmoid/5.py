#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""param_gen.py: The class that converts the hidden vector of the controller into the parameters of the interface
"""
__author__ = " Ryan L. McAvoy"
import torch
from torch.nn import Module


class Param_Generator(Module):
    
    def forward(self, vals):

        # e_t^i - Amount to erase the memory by before writing, for each write head.
        # [batch, num_writes*word_size]
        erase_vec = torch.nn.functional.sigmoid(self.erase_vect_(vals))
        update_data['erase_vectors'] = erase_vec.view(
            -1, self._num_writes, self._word_size)

        # f_t^j - Amount that the memory at the locations read from at the previous
        # time step can be declared unused, for each read head `j`.
        update_data['free_gate'] = torch.nn.functional.sigmoid(
            self.free_gate_(vals)).view(-1, self._num_reads, 1)

        # g_t^{a, i} - Interpolation between writing to unallocated memory and
        # content-based lookup, for each write head `i`. Note: `a` is simply used to
        # identify this gate with allocation vs writing (as defined below).
        update_data['allocation_gate'] = torch.nn.functional.sigmoid(
            self.allocate_gate_(vals)).view(-1, self._num_writes, 1)

        # g_t^{w, i} - Overall gating of write amount for each write head.
        update_data['write_gate'] = torch.nn.functional.sigmoid(
            self.write_gate_(vals)).view(-1, self._num_writes, 1)

        # \pi_t^j - Mixing between "backwards" and "forwards" positions (for
        # each write head), and content-based lookup, for each read head.
        # Need to apply softmax batch-wise to the second index. This will not
        # work
       
        update_data['read_mode_shift'] = torch.nn.functional.sigmoid(
            self.free_gate_(vals)).view(-1, self._num_reads, 1)

        return update_data
