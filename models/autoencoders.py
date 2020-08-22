import torch
from torch import nn

class autoenc1(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 8, kernel_size = 3, stride = 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3, stride = 2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3, stride = 2)
        self.bn3 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU6(inplace=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv1 = nn.Conv2d(32, 16, kernel_size = 1)
        self.deconv2 = nn.Conv2d(16, 8, kernel_size = 1)
        self.deconv3 = nn.Conv2d(8, 3, kernel_size = 1)

    def forward(x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)

        x = self.upsample(x)
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.upsample(x)
        x = self.deconv3(x)
        x = self.relu(x)

        return x
