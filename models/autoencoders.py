import torch
from torch import nn

class autoenc1(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d()
