import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os

# Define the Wide Residual Unit (WRU)
class WideResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(WideResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.prelu1 = nn.PReLU(num_parameters=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.prelu2 = nn.PReLU(num_parameters=out_channels)
        # Skip connection convolution to match dimensions
        self.skip_conv = nn.Conv2d(skip_channels, out_channels, kernel_size=1, padding=0, bias=True) if skip_channels != out_channels else nn.Identity()
        # Identity convolution to match input channels to output channels for residual connection
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True) if in_channels != out_channels else nn.Identity()

    def forward(self, x, skip_input):
        identity = x
        out = self.prelu1(self.conv1(x))
        out = self.conv2(out)
        skip_out = self.skip_conv(skip_input)
        out = out + skip_out
        out = self.prelu2(out)
        # Adjust identity channels to match out
        identity = self.identity_conv(identity)
        out = out + identity
        return out

# Define the Wide Residual Block (WRB)
class WideResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, num_units=3):
        super(WideResidualBlock, self).__init__()
        self.units = nn.ModuleList([
            WideResidualUnit(in_channels if i == 0 else out_channels, out_channels, skip_channels)
            for i in range(num_units)
        ])

    def forward(self, x, skip_input):
        out = x
        for unit in self.units:
            out = unit(out, skip_input)
        return out

# Define the FSCWN Model
class FSCWN(nn.Module):
    def __init__(self):
        super(FSCWN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=True)
        self.prelu1 = nn.PReLU(num_parameters=128)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True)
        self.prelu2 = nn.PReLU(num_parameters=64)
        self.wrb1 = WideResidualBlock(64, 64, 64, num_units=3)
        self.wrb2 = WideResidualBlock(64, 128, 64, num_units=3)
        self.wrb3 = WideResidualBlock(128, 256, 64, num_units=3)
        self.deconv = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.prelu_deconv = nn.PReLU(num_parameters=128)
        self.conv_recon1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.prelu_recon1 = nn.PReLU(num_parameters=128)
        self.conv_recon2 = nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=True)
        self.global_skip = nn.Identity()

    def forward(self, x):
        # Input shape: [batch_size, 1, H, W] (e.g., [8, 1, 64, 64])
        # print(f'Input: {x.shape}')
        identity = x

        # conv1: 1 -> 128 channels, same spatial size
        # Shape: [batch_size, 128, H, W] (e.g., [8, 128, 64, 64])
        out = self.prelu1(self.conv1(x))
        # print(f'After conv1: {out.shape}')

        # pool: Reduce spatial size by half, no channel change
        # Shape: [batch_size, 128, H/2, W/2] (e.g., [8, 128, 32, 32])
        out = self.pool(out)
        # print(f'After pool: {out.shape}')

        # conv2: 128 -> 64 channels, same spatial size
        # Shape: [batch_size, 64, H/2, W/2] (e.g., [8, 64, 32, 32])
        skip_input = self.prelu2(self.conv2(out))
        # print(f'After conv2 (skip_input): {skip_input.shape}')

        # Set out to skip_input for wrb1 to ensure 64 channels
        # Shape: [batch_size, 64, H/2, W/2] (e.g., [8, 64, 32, 32])
        out = skip_input
        # print(f'Before wrb1: {out.shape}')

        # wrb1: 64 -> 64 channels, same spatial size
        # Shape: [batch_size, 64, H/2, W/2] (e.g., [8, 64, 32, 32])
        out = self.wrb1(out, skip_input)
        # print(f'After wrb1: {out.shape}')

        # wrb2: 64 -> 128 channels, same spatial size
        # Shape: [batch_size, 128, H/2, W/2] (e.g., [8, 128, 32, 32])
        out = self.wrb2(out, skip_input)
        # print(f'After wrb2: {out.shape}')

        # wrb3: 128 -> 256 channels, same spatial size
        # Shape: [batch_size, 256, H/2, W/2] (e.g., [8, 256, 32, 32])
        out = self.wrb3(out, skip_input)
        # print(f'After wrb3: {out.shape}')

        # deconv: 256 -> 128 channels, upsample spatial size by 2
        # Shape: [batch_size, 128, H, W] (e.g., [8, 128, 64, 64])
        out = self.prelu_deconv(self.deconv(out))
        # print(f'After deconv: {out.shape}')

        # conv_recon1: 128 -> 128 channels, same spatial size
        # Shape: [batch_size, 128, H, W] (e.g., [8, 128, 64, 64])
        out = self.prelu_recon1(self.conv_recon1(out))
        # print(f'After conv_recon1: {out.shape}')

        # conv_recon2: 128 -> 1 channel, same spatial size
        # Shape: [batch_size, 1, H, W] (e.g., [8, 1, 64, 64])
        out = self.conv_recon2(out)
        # print(f'After conv_recon2: {out.shape}')

        # Global residual: Add input
        # Shape: [batch_size, 1, H, W] (e.g., [8, 1, 64, 64])
        out = out + identity
        # print(f'Output: {out.shape}')

        return out