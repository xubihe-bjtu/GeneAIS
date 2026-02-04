import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ComplexReLU(nn.Module):
    """复数ReLU激活函数"""

    def __init__(self):
        super(ComplexReLU, self).__init__()

    def forward(self, x):
        # 对实部和虚部分别应用ReLU
        real = F.relu(x.real)
        imag = F.relu(x.imag)
        return torch.complex(real, imag)


class ComplexLayerNorm(nn.Module):
    """复数层归一化"""

    def __init__(self, normalized_shape, eps=1e-5):
        super(ComplexLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight_real = nn.Parameter(torch.ones(normalized_shape))
        self.weight_imag = nn.Parameter(torch.ones(normalized_shape))
        self.bias_real = nn.Parameter(torch.zeros(normalized_shape))
        self.bias_imag = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # 对实部和虚部分别进行层归一化
        real = F.layer_norm(x.real, self.normalized_shape, eps=self.eps)
        imag = F.layer_norm(x.imag, self.normalized_shape, eps=self.eps)

        real = real * self.weight_real + self.bias_real
        imag = imag * self.weight_imag + self.bias_imag

        return torch.complex(real, imag)


class ComplexSigmoid(nn.Module):
    """复数Sigmoid激活函数"""

    def __init__(self):
        super(ComplexSigmoid, self).__init__()

    def forward(self, x):
        # 对实部和虚部分别应用Sigmoid
        real = torch.sigmoid(x.real)
        imag = torch.sigmoid(x.imag)
        return torch.complex(real, imag)

class ComplexDropout(nn.Module):
    """复数Dropout"""

    def __init__(self, p=0.1):
        super(ComplexDropout, self).__init__()
        self.p = p
        self.dropout_real = nn.Dropout(p)
        self.dropout_imag = nn.Dropout(p)

    def forward(self, x):
        real = self.dropout_real(x.real)
        imag = self.dropout_imag(x.imag)
        return torch.complex(real, imag)