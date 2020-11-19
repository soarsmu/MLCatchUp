import torch
from torch import nn
from torch.autograd import Variable

class WaveNet(nn.Module):
   
    def residual_forward(self, _input, conv_sigmoid, conv_tanh, skip_scale,
                         residue_scale):
        output = _input

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
