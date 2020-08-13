import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules import conv
from torch.nn import functional as F
from torch.nn.modules.utils import _triple
from torch.nn.functional import conv3d


class ConvTTN3d_old(conv._ConvNd):
    # original version of the __init__
    def __init__(self, in_channels, out_channels, kernel_size, project_variable, transformation_groups, k0_groups,
                    transformations_per_filter, stride=1, padding=0, dilation=1, groups=1, bias=True,
                    padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        super(ConvTTN3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode)

        self.project_variable = project_variable

        assert (0 < transformation_groups <= out_channels)
        self.transformation_groups = transformation_groups

        assert (0 < k0_groups <= out_channels)
        self.k0_groups = k0_groups

        # either we learn one unique transform for each temporal slice pair OR
        # we learn just a single transform shared by all temporal slices in one filter
        assert (transformations_per_filter == kernel_size[0] - 1 or transformations_per_filter == 1)
        self.transformations_per_filter = transformations_per_filter

        # replaced out_channels with k0_groups
        first_w = torch.nn.Parameter(torch.zeros(self.k0_groups, in_channels, 1, kernel_size[1], kernel_size[2]))

        # when transfer learning this will be an issue? set k0_init to None when transfer learning
        if project_variable.k0_init == 'normal':
            self.first_weight = torch.nn.init.normal_(first_w)
        elif project_variable.k0_init == 'ones':
            self.first_weight = torch.nn.init.constant(first_w, 1)
        elif project_variable.k0_init == 'ones_var':
            self.first_weight = torch.nn.init.normal_(first_w, mean=1., std=0.5)
        elif project_variable.k0_init == 'uniform':
            self.first_weight = torch.nn.init.uniform(first_w)
        elif project_variable.k0_init == 'kaiming-uniform':
            self.first_weight = torch.nn.init.kaiming_uniform_(first_w)
        elif project_variable.k0_init == 'kaiming-normal':
            self.first_weight = torch.nn.init.kaiming_normal_(first_w)

        if self.project_variable.theta_init is not None:
            # self.theta = torch.zeros((kernel_size[0] - 1, out_channels, 2, 3))
            # self.theta = torch.zeros((kernel_size[0] - 1, self.transformation_groups, 2, 3))
            self.theta = torch.zeros((self.transformations_per_filter, self.transformation_groups, 2, 3))
            if self.project_variable.theta_init == 'eye':
                for i in range(self.transformations_per_filter):
                    for j in range(self.transformation_groups):
                        self.theta[i][j] = torch.eye(3)[:2, ]

            elif self.project_variable.theta_init == 'normal':
                self.theta = torch.abs(torch.nn.init.normal_(self.theta))

            elif self.project_variable.theta_init == 'eye-like':
                for i in range(self.transformations_per_filter):
                    for j in range(self.transformation_groups):
                        self.theta[i][j] = torch.eye(3)[:2, ] + torch.nn.init.normal_(torch.zeros(2, 3), mean=0,
                                                                                      std=1e-5)

            else:
                print("ERROR: theta_init mode '%s' not supported" % self.project_variable.theta_init)
                self.theta = None

            self.theta = torch.nn.Parameter(self.theta)

        # replaced 'out_channels' with 'transformation_groups'
        # replaced 'kernel_size[0] - 1' with 'transformations_per_filter'
        else:
            if self.project_variable.srxy_init == 'normal':
                # use 4 parameters
                self.scale = torch.nn.Parameter(
                    torch.abs(torch.nn.init.normal_(
                        torch.zeros((self.transformations_per_filter, self.transformation_groups)))))
                self.rotate = torch.nn.init.normal_(
                    torch.nn.Parameter(torch.zeros((self.transformations_per_filter, self.transformation_groups))))
                self.translate_x = torch.nn.init.normal_(
                    torch.nn.Parameter(torch.zeros((self.transformations_per_filter, self.transformation_groups))))
                self.translate_y = torch.nn.init.normal_(
                    torch.nn.Parameter(torch.zeros((self.transformations_per_filter, self.transformation_groups))))
            elif self.project_variable.srxy_init == 'eye':
                self.scale = torch.nn.Parameter(
                    torch.nn.init.ones_(torch.zeros((self.transformations_per_filter, self.transformation_groups))))
                self.rotate = torch.nn.Parameter(
                    torch.zeros((self.transformations_per_filter, self.transformation_groups)))
                self.translate_x = torch.nn.Parameter(
                    torch.zeros((self.transformations_per_filter, self.transformation_groups)))
                self.translate_y = torch.nn.Parameter(
                    torch.zeros((self.transformations_per_filter, self.transformation_groups)))
            elif self.project_variable.srxy_init == 'eye-like':
                self.scale = torch.nn.Parameter(torch.abs(
                    torch.nn.init.normal_(torch.zeros((self.transformations_per_filter, self.transformation_groups)),
                                          mean=1, std=1e-5)))
                self.rotate = torch.nn.init.normal_(
                    torch.nn.Parameter(torch.zeros((self.transformations_per_filter, self.transformation_groups))),
                    mean=0, std=1e-5)
                self.translate_x = torch.nn.init.normal_(
                    torch.nn.Parameter(torch.zeros((self.transformations_per_filter, self.transformation_groups))),
                    mean=0, std=1e-5)
                self.translate_y = torch.nn.init.normal_(
                    torch.nn.Parameter(torch.zeros((self.transformations_per_filter, self.transformation_groups))),
                    mean=0, std=1e-5)
            else:
                print("ERROR: srxy_init mode '%s' not supported" % self.project_variable.srxy_init)
                self.scale, self.rotate, self.translate_x, self.translate_y = None, None, None, None


    def make_affine_matrix(self, scale, rotate, translate_x, translate_y):
        # if out_channels is used, the shape of the matrix returned is different

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

    # replace out_channels with self.transformation_groups
    def update_2(self, grid, theta, device):
        # deal with updating s r x y

        if theta is not None:

            for i in range(self.transformations_per_filter):
                tmp = self.make_affine_matrix(self.scale[i], self.rotate[i], self.translate_x[i], self.translate_y[i])
                tmp = tmp.cuda(device)
                theta = torch.cat((theta, tmp.unsqueeze(0)), 0)
            theta = theta[1:]

            try:
                _ = F.affine_grid(theta[0],
                                  [self.transformation_groups, self.transformations_per_filter, self.kernel_size[1],
                                   self.kernel_size[2]], align_corners=True)
            except RuntimeError:
                torch.backends.cudnn.deterministic = True
                print('ok cudnn')

            if self.transformations_per_filter == self.kernel_size[0] - 1:
                for i in range(self.kernel_size[0] - 1):
                    tmp = F.affine_grid(theta[i],
                                        [self.transformation_groups, self.kernel_size[0], self.kernel_size[1],
                                         self.kernel_size[2]], align_corners=True)
                    grid = torch.cat((grid, tmp.unsqueeze(0)), 0)
            else:
                tmp = F.affine_grid(theta[0],
                                    [self.transformation_groups, self.kernel_size[0], self.kernel_size[1],
                                     self.kernel_size[2]], align_corners=True)
                for i in range(self.kernel_size[0] - 1):
                    grid = torch.cat((grid, tmp.unsqueeze(0)), 0)

            return grid

        else:
            # cudnn error
            try:
                _ = F.affine_grid(self.theta[0],
                                  [self.transformation_groups, self.transformations_per_filter, self.kernel_size[1],
                                   self.kernel_size[2]], align_corners=True)
            except RuntimeError:
                torch.backends.cudnn.deterministic = True
                print('ok cudnn')

            if self.transformations_per_filter == self.kernel_size[0] - 1:
                for i in range(self.kernel_size[0] - 1):
                    tmp = F.affine_grid(self.theta[i],
                                        [self.transformation_groups, self.kernel_size[0], self.kernel_size[1],
                                         self.kernel_size[2]], align_corners=True)
                    grid = torch.cat((grid, tmp.unsqueeze(0)), 0)
            else:
                tmp = F.affine_grid(self.theta[0],
                                    [self.transformation_groups, self.kernel_size[0], self.kernel_size[1],
                                     self.kernel_size[2]], align_corners=True)

                for i in range(self.kernel_size[0] - 1):
                    grid = torch.cat((grid, tmp.unsqueeze(0)), 0)

            return grid

    # replace out_channels with transformation_groups
    def forward(self, input, device):

        grid = torch.zeros((1, self.transformation_groups, self.kernel_size[1], self.kernel_size[2], 2))

        if self.project_variable.theta_init is None:
            # add smoothness constraint for SRXY
            if self.project_variable.srxy_smoothness == 'sigmoid':
                self.scale.data = (2 - 0) * torch.nn.functional.sigmoid(-self.scale) + 0  # between 0 and 2
                self.rotate.data = (1 - -1) * torch.nn.functional.sigmoid(-self.rotate) - 1  # between -1 and 1
                self.translate_x.data = (1 - -1) * torch.nn.functional.sigmoid(
                    -self.translate_x) - 1  # between -1 and 1
                self.translate_y.data = (1 - -1) * torch.nn.functional.sigmoid(
                    -self.translate_y) - 1  # between -1 and 1
            elif self.project_variable.srxy_smoothness == 'sigmoid_small':
                self.scale.data = (1.1 - 0.9) * torch.nn.functional.sigmoid(-self.scale) + 0.9  # between 0.9 and 1.1
                self.rotate.data = (0.5 - -0.5) * torch.nn.functional.sigmoid(
                    -self.rotate) - 0.5  # between -0.5 and 0.5
                self.translate_x.data = (0.25 - -0.25) * torch.nn.functional.sigmoid(
                    -self.translate_x) - 0.25  # between -0.25 and 0.25
                self.translate_y.data = (0.25 - -0.25) * torch.nn.functional.sigmoid(
                    -self.translate_y) - 0.25  # between -0.25 and 0.25

            theta = torch.zeros((1, self.transformation_groups, 2, 3))
            theta = theta.cuda(device)
        else:
            theta = None

        # if PV.theta_init is None, then theta is not None -> no 'eye'
        # if PV.theta_init is not None, then theta is None -> 'eye' is used

        grid = grid.cuda(device)
        grid = self.update_2(grid, theta, device)
        grid = grid[1:]

        # check if shape of grid is compatible with out_channels. if not, correct it
        if self.out_channels != grid.shape[1]:
            if self.out_channels % self.transformation_groups == 0:
                grid = grid.repeat_interleave(self.out_channels // self.transformation_groups, 1)
            else:
                grid = grid.repeat_interleave(self.out_channels // self.transformation_groups + 1, 1)
                grid = grid[:, :self.out_channels, :, :, :]

        # check if first_weight is compatible with out_channels. if not, correct it
        if self.out_channels % self.k0_groups == 0:
            times = self.out_channels // self.k0_groups
        else:
            times = self.out_channels // self.k0_groups + 1

        # ---
        # needed to deal with the cudnn error
        try:
            _ = F.grid_sample(self.first_weight[:, :, 0].repeat_interleave(times, 0)[:self.out_channels], grid[0],
                              mode='bilinear', padding_mode='zeros', align_corners=True)
            # _ = F.grid_sample(self.first_weight[:, :, 0], grid[0])
        except RuntimeError:
            torch.backends.cudnn.deterministic = True
            print('ok cudnn')
        # ---

        new_weight = self.first_weight.repeat_interleave(times, 0)[:self.out_channels]

        if self.transformations_per_filter == self.kernel_size[0] - 1:
            if self.project_variable.weight_transform == 'naive':
                for i in range(self.kernel_size[0] - 1):
                    tmp = F.grid_sample(self.first_weight[:, :, 0].repeat_interleave(times, 0)[:self.out_channels],
                                        grid[i], mode='bilinear', padding_mode='zeros', align_corners=True)
                    new_weight = torch.cat((new_weight, tmp.unsqueeze(2)), 2)
            elif self.project_variable.weight_transform == 'seq':
                for i in range(self.kernel_size[0] - 1):
                    tmp = F.grid_sample(new_weight[:, :, -1], grid[i], mode='bilinear', padding_mode='zeros',
                                        align_corners=True)
                    new_weight = torch.cat((new_weight, tmp.unsqueeze(2)), 2)
            else:
                print('ERROR: weight_transform with value %s not supported' % self.project_variable.weight_transform)
        else:
            if self.project_variable.weight_transform == 'naive':
                tmp = F.grid_sample(self.first_weight[:, :, 0].repeat_interleave(times, 0)[:self.out_channels], grid[0],
                                    mode='bilinear', padding_mode='zeros', align_corners=True)
                for i in range(self.kernel_size[0] - 1):
                    new_weight = torch.cat((new_weight, tmp.unsqueeze(2)), 2)

            elif self.project_variable.weight_transform == 'seq':
                tmp = F.grid_sample(new_weight[:, :, -1], grid[0], mode='bilinear', padding_mode='zeros',
                                    align_corners=True)
                for i in range(self.kernel_size[0] - 1):
                    new_weight = torch.cat((new_weight, tmp.unsqueeze(2)), 2)
            else:
                print('ERROR: weight_transform with value %s not supported' % self.project_variable.weight_transform)

        self.weight = torch.nn.Parameter(new_weight)

        y = F.conv3d(input, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y


# simplified version of ConvTTN3d class
# removed: transformation_groups, k0_groups
# default: k0_init = 'normal', theta_init = 'eye',
class ConvTTN3d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, project_variable, transformation_groups=None,
                 k0_groups=None, transformations_per_filter=None, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        super(ConvTTN3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode)

        self.project_variable = project_variable

        # replaced with k0_groups with out_channels
        first_w = torch.nn.Parameter(torch.zeros(self.out_channels, in_channels, 1, kernel_size[1], kernel_size[2]))

        # default: k0_init = 'normal'
        self.first_weight = torch.nn.init.normal_(first_w)

        # default: theta_init = 'eye'
        # replace transformation_groups with out_channels
        # replace transformations_per_filter with kernel_size[0]-1
        # default: srxy_init = 'eye
        self.scale = torch.nn.Parameter(
            torch.nn.init.ones_(torch.zeros((self.kernel_size[0]-1, self.out_channels))))
        self.rotate = torch.nn.Parameter(
            torch.zeros((self.kernel_size[0]-1, self.out_channels)))
        self.translate_x = torch.nn.Parameter(
            torch.zeros((self.kernel_size[0]-1, self.out_channels)))
        self.translate_y = torch.nn.Parameter(
            torch.zeros((self.kernel_size[0]-1, self.out_channels)))

    def make_affine_matrix(self, scale, rotate, translate_x, translate_y):
        # if out_channels is used, the shape of the matrix returned is different

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

    # replace out_channels with self.transformation_groups <- remove in a bit
    # replace transformation_groups with out_channels
    # replace transformations_per_filter with kernel_size[0]-1

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
    def forward(self, input, device):
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

        y = F.conv3d(input, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y
