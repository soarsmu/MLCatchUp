import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules import conv
from torch.nn import functional as F
from torch.nn.modules.utils import _triple
from torch.nn.functional import conv3d


# version of 3tconv where the dimensions are variable.


class ConvTTN3d_dynamic(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, project_variable, transformation_groups=None,
                 k0_groups=None, transformations_per_filter=None, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        assert len(kernel_size) == 3 # (d, h, w)
        # kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        super(ConvTTN3d_dynamic, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode)

        self.project_variable = project_variable

        first_w = torch.nn.Parameter(torch.zeros(self.out_channels, in_channels, 1, kernel_size[1], kernel_size[2]))

        self.first_weight = torch.nn.init.normal_(first_w)

        self.scale = torch.nn.Parameter(
            torch.nn.init.ones_(torch.zeros((self.kernel_size[0]-1, self.out_channels))))
        self.rotate = torch.nn.Parameter(
            torch.zeros((self.kernel_size[0]-1, self.out_channels)))
        self.translate_x = torch.nn.Parameter(
            torch.zeros((self.kernel_size[0]-1, self.out_channels)))
        self.translate_y = torch.nn.Parameter(
            torch.zeros((self.kernel_size[0]-1, self.out_channels)))

    def make_affine_matrix(self, scale, rotate, translate_x, translate_y):

        assert scale.shape == rotate.shape == translate_x.shape == translate_y.shape

        '''
        matrix.shape = (out_channels, 2, 3)
        '''
        matrix = torch.zeros((scale.shape[0], 2, 3))

        matrix[:, 0, 0] = scale[:] * torch.cos(rotate[:])
        matrix[:, 0, 1] = -scale[:] * torch.sin(rotate[:])
        matrix[:, 0, 2] = translate_x[:] * scale[:] * torch.cos(rotate[:]) - translate_y[:] * \
                          scale[:] * torch.sin(rotate[:])
        matrix[:, 1, 0] = scale[:] * torch.sin(rotate[:])
        matrix[:, 1, 1] = scale[:] * torch.cos(rotate[:])
        matrix[:, 1, 2] = translate_x[:] * scale[:] * torch.sin(rotate[:]) + translate_y[:] * \
                          scale[:] * torch.cos(rotate[:])

        return matrix


    def update_2(self, grid, theta, device):
        # deal with updating s r x y

        # if theta is not None: (theta is not None because we use srxy starting from eye)

        for i in range(self.kernel_size[0]-1):
            tmp = self.make_affine_matrix(self.scale[i], self.rotate[i], self.translate_x[i], self.translate_y[i])
            tmp = tmp.cuda(device)
            theta = torch.cat((theta, tmp.unsqueeze(0)), 0)
        theta = theta[1:]

        try:
            if torch.__version__ == '1.2.0':
                _ = F.affine_grid(theta[0],
                                  [self.out_channels, self.kernel_size[0] - 1, self.kernel_size[1],
                                   self.kernel_size[2]])
            else:
                _ = F.affine_grid(theta[0],
                                  [self.out_channels, self.kernel_size[0]-1, self.kernel_size[1],
                                   self.kernel_size[2]], align_corners=True)
        except RuntimeError:
            torch.backends.cudnn.deterministic = True
            print('ok cudnn')

        for i in range(self.kernel_size[0] - 1):
            if torch.__version__ == '1.2.0':
                tmp = F.affine_grid(theta[i],
                                    [self.out_channels, self.kernel_size[0], self.kernel_size[1],
                                     self.kernel_size[2]])
            else:
                tmp = F.affine_grid(theta[i],
                                    [self.out_channels, self.kernel_size[0], self.kernel_size[1],
                                     self.kernel_size[2]], align_corners=True)

            grid = torch.cat((grid, tmp.unsqueeze(0)), 0)

        return grid

    # replace out_channels with transformation_groups <- remove in a bit
    def forward(self, the_input, device):
        # replace transformation_groups with out_channels
        # replace transformations_per_filter with kernel_size[0]-1

        grid = torch.zeros((1, self.out_channels, self.kernel_size[1], self.kernel_size[2], 2))

        # default: theta_init = None (then theta is created from srxy parameters)
        # default: srxy_smoothness = None
        theta = torch.zeros((1, self.out_channels, 2, 3))
        theta = theta.cuda(device)

        grid = grid.cuda(device)
        grid = self.update_2(grid, theta, device)
        grid = grid[1:]

        # default: transformation_group = out_channels
        # default: k0_groups = out_channels
        # default: transformation_per_filter = kernel_size[0]-1

        # ---
        # needed to deal with the cudnn error
        try:
            if torch.__version__ == '1.2.0':
                _ = F.grid_sample(self.first_weight[:, :, 0], grid[0], mode='bilinear', padding_mode='zeros')
            else:
                _ = F.grid_sample(self.first_weight[:, :, 0], grid[0], mode='bilinear', padding_mode='zeros',
                                  align_corners=True)
        except RuntimeError:
            torch.backends.cudnn.deterministic = True
            print('ok cudnn')
        # ---

        new_weight = self.first_weight

        # default: weight_transform = 'seq'
        for i in range(self.kernel_size[0] - 1):
            if torch.__version__ == '1.2.0':
                tmp = F.grid_sample(new_weight[:, :, -1], grid[i], mode='bilinear', padding_mode='zeros')
            else:
                tmp = F.grid_sample(new_weight[:, :, -1], grid[i], mode='bilinear', padding_mode='zeros',
                                    align_corners=True)
            new_weight = torch.cat((new_weight, tmp.unsqueeze(2)), 2)

        self.weight = torch.nn.Parameter(new_weight)

        y = F.conv3d(the_input, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y
