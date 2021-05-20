import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple

# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


class ScaledStdConv2dSame(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization and Tensorflow-like SAME padding support
    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692
    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding='SAME', dilation=1, groups=1,
            bias=True, gamma=1.0, eps=1e-5, gain_init=1.0, use_layernorm=False):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.same_pad = is_dynamic
        self.eps = eps ** 2 if use_layernorm else eps
        self.use_layernorm = use_layernorm  # experimental, slightly faster/less GPU memory to hijack LN kernel

    # NOTE an alternate formulation to consider, closer to DeepMind Haiku impl but doesn't seem
    # to make much numerical difference (+/- .002 to .004) in top-1 during eval.
    def get_weight(self):
        var, mean = torch.var_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        scale = torch.rsqrt((self.weight[0].numel() * var).clamp_(self.eps)) * self.gain
        weight = (self.weight - mean) * scale
        return self.gain * weight

    # def get_weight(self):
    #     if self.use_layernorm:
    #         weight = self.scale * F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
    #     else:
    #         std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
    #         weight = self.scale * (self.weight - mean) / (std + self.eps)
    #     return self.gain * weight

    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)