import torch
from torch import nn

class Conv_block(nn.Sequential):
    def __init__(self, conv_type, in_channels, out_channels):
        if conv_type == 1:
            super(Conv_block, self).__init__(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size = 1,
                    stride = 1
                )
            )
class Inception_block(nn.Module):
    def __init__(self):
        super(Inception_block, self).__init__()
