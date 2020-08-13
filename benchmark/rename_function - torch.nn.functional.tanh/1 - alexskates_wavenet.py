import torch
from torch import nn
from torch.autograd import Variable


class One_Hot(nn.Module):
    # Functionality to discretise the data
    # Apparently better results than continuous data
    def __init__(self, size):
        super(One_Hot, self).__init__()
        self.size = size
        self.ones = torch.sparse.torch.eye(size)

    def forward(self, x_in):
        return Variable(self.ones.index_select(0, x_in.data))

    def __repr__(self):
        # Enable you to print out the value of the class
        return self.__class__.__name__ + "({})".format(self.depth)


class WaveNet(nn.Module):
    def __init__(self, n_out=256, n_residue=32, n_skip=512, dilation_depth=10,
                 n_layers=5):
        # n_out: size of output
        # n_residue: size of residue channels
        # n_skip: size of skip channels
        # dilation_depth & n_layers: dilation layer setup
        super(WaveNet, self).__init__()

        self.dilation_depth = dilation_depth
        dilations = self.dilations = [2 ** i for i in
                                      range(dilation_depth)] * n_layers

        self.one_hot = One_Hot(n_out)

        self.from_input = nn.Conv1d(in_channels=n_out, out_channels=n_residue,
                                    kernel_size=1)

        self.conv_sigmoid = nn.ModuleList(
            [nn.Conv1d(in_channels=n_residue, out_channels=n_residue,
                       kernel_size=2, dilation=d) for d in dilations])
        self.conv_tanh = nn.ModuleList(
            [nn.Conv1d(in_channels=n_residue, out_channels=n_residue,
                       kernel_size=2, dilation=d) for d in dilations])
        self.skip_scale = nn.ModuleList(
            [nn.Conv1d(in_channels=n_residue, out_channels=n_skip,
                       kernel_size=1) for _ in dilations])
        self.residue_scale = nn.ModuleList(
            [nn.Conv1d(in_channels=n_residue, out_channels=n_residue,
                       kernel_size=1) for _ in dilations])

        self.conv_post_1 = nn.Conv1d(in_channels=n_skip, out_channels=n_skip,
                                     kernel_size=1)
        self.conv_post_2 = nn.Conv1d(in_channels=n_skip, out_channels=n_out,
                                     kernel_size=1)

    def forward(self, _input):
        output = self.preprocess(_input)

        skip_connections = []
        # iterate over each residual block
        for s, t, skip_scale, residue_scale in zip(
                self.conv_sigmoid,
                self.conv_tanh,
                self.skip_scale,
                self.residue_scale):
            output, skip = self.residual_forward(
                output, s, t, skip_scale, residue_scale)
            skip_connections.append(skip)

        # sum up skip connections
        output = sum([s[:, :, -output.size(2):] for s in skip_connections])
        output = self.postprocess(output)
        return output

    def preprocess(self, _input):
        output = self.one_hot(_input).unsqueeze(0).transpose(1, 2)
        output = self.from_input(output)
        return output

    def postprocess(self, _input):
        output = nn.functional.elu(_input)  # Supposedly better than ReLU
        output = self.conv_post_1(output)
        output = nn.functional.elu(output)
        output = self.conv_post_2(output).squeeze(0).transpose(0, 1)
        return output

    def residual_forward(self, _input, conv_sigmoid, conv_tanh, skip_scale,
                         residue_scale):
        output = _input

        output_sigmoid, output_tanh = conv_sigmoid(output), conv_tanh(output)
        # Multiply them together
        output = nn.functional.sigmoid(output_sigmoid) * nn.functional.tanh(
            output_tanh)

        # Appears to be only one 1x1 conv in the paper, but code shows two
        skip = skip_scale(output)  # 1x1 convolution, this to skip connections
        output = residue_scale(output)  # 1x1 convolution, this to next residual block

        output = output + _input[:, :, -output.size(2):]
        return output, skip

    def generate(self, _input, n=100):
        res = _input.data.tolist()

        for _ in range(n):
            x = Variable(torch.LongTensor(res[-sum(self.dilations) - 1:]))
            y = self.forward(x)
            _, i = y.max(dim=1)

            res.append(i.data.tolist()[-1])
        return res
