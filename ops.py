import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils
import math

class conv5x5(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(conv5x5, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride,
                              padding=2*dilation, dilation=dilation, bias=False)
        self.conv = utils.spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)

class conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                              padding=dilation, dilation=dilation, padding_mode='reflect', bias=False)
        self.conv = utils.spectral_norm(self.conv)
    def forward(self, x):
        return self.conv(x)

class conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.conv = utils.spectral_norm(self.conv)
    def forward(self, x):
        return self.conv(x)

class conv_zeros(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_zeros, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.constant_(self.conv.weight, 0)

    def forward(self, x):
        return self.conv(x)

class PAKA3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(PAKA3x3, self).__init__()
        self.conv = PAKA2d(in_channels, out_channels, kernel_size=3, stride=stride,
                              padding=dilation, dilation=dilation, bias=False)
        self.conv = utils.spectral_norm(self.conv)
    def forward(self, x):
        return self.conv(x)

class PAKA2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(PAKA2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size ** 2, 1, 1))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.conv_c = nn.Sequential(conv1x1(in_channels, in_channels, stride),
                                    nn.ReLU(True),
                                    conv_zeros(in_channels, in_channels),
                                    )

        self.conv_d = nn.Sequential(conv3x3(in_channels, in_channels, stride, dilation=dilation),
                                    nn.ReLU(True),
                                    conv_zeros(in_channels, kernel_size ** 2),
                                    )

        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride, dilation=dilation)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def forward(self, x):
        b, n, h, w = x.shape
        return F.conv3d(self.unfold(x).view(b, n, self.kernel_size ** 2, h//self.stride, w//self.stride) * (1 + torch.tanh(self.conv_d(x).unsqueeze(1)+self.conv_c(x).unsqueeze(2))),
                        self.weight, self.bias).squeeze(2)

class downsample(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(downsample, self).__init__()

        self.conv1 = conv3x3(in_channels, hidden_channels)
        self.conv2 = conv3x3(hidden_channels, out_channels, stride=2)

    def forward(self, x):
        h = self.conv1(x)
        h = F.elu(h)
        h = self.conv2(h)
        h = F.elu(h)
        return h

class upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels*4)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        h = self.conv1(x)
        h = F.pixel_shuffle(h, 2)
        h = F.elu(h)
        h = self.conv2(h)
        h = F.elu(h)
        return h
