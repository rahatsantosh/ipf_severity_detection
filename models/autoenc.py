import torch
from torch import nn

def get_same_padding(kernel_size):
    """Calculate padding size for same padding,
    assuming stride of 1 and square kernel"""

    if type(kernel_size) is tuple:
        kernel_size = kernel_size[0]
    pad_size = (kernel_size-1)//2
    if kernel_size%2 == 0:
        padding = (pad_size, pad_size+1)
    else:
        padding = pad_size

    return padding


class Conv_block(nn.Sequential):
    def __init__(self, conv_type, in_channels, out_channels, inter_channels = None):
        if conv_type == "1x1":
            super(Conv_block, self).__init__(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size = 1,
                    stride = 1,
                    padding = get_same_padding(1)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        elif conv_type == "3x3":
            assert inter_channels is not None
            super(Conv_block, self).__init__(
                nn.Conv2d(
                    in_channels,
                    inter_channels,
                    kernel_size = 1,
                    stride = 1,
                    padding = get_same_padding(1)
                ),
                nn.ReLU6(inplace=True),
                nn.Conv2d(
                    inter_channels,
                    out_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = get_same_padding(3)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        else:
            assert inter_channels is not None
            super(Conv_block, self).__init__(
                nn.Conv2d(
                    in_channels,
                    inter_channels,
                    kernel_size = 1,
                    stride = 1,
                    padding = get_same_padding(1)
                ),
                nn.ReLU6(inplace=True),
                nn.Conv2d(
                    inter_channels,
                    out_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = get_same_padding(3)
                ),
                nn.ReLU6(inplace=True),
                nn.Conv2d(
                    inter_channels,
                    out_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = get_same_padding(3)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )

class Deconv_block(nn.Sequential):
    def __init__(self, conv_type, in_channels, out_channels, inter_channels = None):
        if conv_type == "1x1":
            super(Deconv_block, self).__init__(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size = 1,
                    stride = 1,
                    padding = get_same_padding(1)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        elif conv_type == "3x3":
            assert inter_channels is not None
            super(Deconv_block, self).__init__(
                nn.ConvTranspose2d(
                    in_channels,
                    inter_channels,
                    kernel_size = 1,
                    stride = 1,
                    padding = get_same_padding(1)
                ),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose2d(
                    inter_channels,
                    out_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = get_same_padding(3)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        else:
            assert inter_channels is not None
            super(Deconv_block, self).__init__(
                nn.ConvTranspose2d(
                    in_channels,
                    inter_channels,
                    kernel_size = 1,
                    stride = 1,
                    padding = get_same_padding(1)
                ),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose2d(
                    inter_channels,
                    out_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = get_same_padding(3)
                ),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose2d(
                    inter_channels,
                    out_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = get_same_padding(3)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )

class Inception_block(nn.Module):
    def __init__(self, in_channels, out_channels, inter3, inter5,
        concat_dim = 2):
        super(Inception_block, self).__init__()

        self.concat_dim = concat_dim
        self.conv1x1 = Conv_block("1x1", in_channels, out_channels, None)
        self.conv3x3 = Conv_block("3x3", in_channels, out_channels, inter3)
        self.conv1x1 = Conv_block("5x5", in_channels, out_channels, inter5)

    def forward(self, x):

        a = self.conv1x1(x)
        b = self.conv3x3(x)
        c = self.conv5x5(x)

        x = torch.cat((a, b, c), dim=self.concat_dim)

        return x


class Inception_block_r(nn.Module):
    def __init__(self, in_channels, out_channels, inter3, inter5,
        concat_dim = 2):
        super(Inception_block_r, self).__init__()

        self.concat_dim = concat_dim
        self.deconv1x1 = Deconv_block("1x1", in_channels, out_channels, None)
        self.deconv3x3 = Deconv_block("3x3", in_channels, out_channels, inter3)
        self.deconv1x1 = Deconv_block("5x5", in_channels, out_channels, inter5)

    def forward(self, x):

        a = self.deconv1x1(x)
        b = self.deconv3x3(x)
        c = self.deconv5x5(x)

        x = torch.cat((a, b, c), dim=self.concat_dim)

        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.inception1 = Inception_block(3, 32, 64, 64)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=2)
        self.inception2 = Inception_block(64, 64, 32, 32)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=2)
        self.inception3 = Inception_block(128, 64, 128, 128)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=2)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.inception1(x)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.inception2(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.inception3(x)
        x = self.conv3(x)
        x = self.relu(x)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.inceptionr1 = Inception_block_r(3, 32, 64, 64)
        self.deconv1 = nn.ConvTranspose2d(32, 64, kernel_size=(3,3), stride=2)
        self.inceptionr2 = Inception_block_r(64, 64, 32, 32)
        self.deconv2 = nn.ConvTranspose2d(64, 128, kernel_size=(3,3), stride=2)
        self.inceptionr3 = Inception_block_r(128, 64, 128, 128)
        self.deconv3 = nn.ConvTranspose2d(64, 128, kernel_size=(3,3), stride=2)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.inceptionr1(x)

        x = self.relu(x)
        x = self.deconv2(x)
        x = self.inceptionr2(x)

        x = self.relu(x)
        x = self.deconv3(x)
        x = self.inceptionr3(x)

        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
