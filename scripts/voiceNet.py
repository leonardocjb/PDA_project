import torch
import torch.nn as nn
from preprocessing import *
from pathlib import Path
from config import *
from speech_dataset import *


class VoiceClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(VoiceClassificationModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x_res = x  # Save the current state for the residual connection
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = x + x_res  # Residual connection
        x = torch.sigmoid(self.fc3(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride)
        self.conv2 = ConvBlock(out_channels, out_channels, stride=1)

        # If the dimensions change, use a 1x1 convolution to match them
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = nn.functional.relu(out)
        return out


# Define the ResNet-9 architecture for voice classification
class voiceNet(nn.Module):
    def __init__(self, input_size=384):
        super(voiceNet, self).__init__()
        self.in_channels = 64
        self.conv1 = ConvBlock(1, 64, stride=2)
        self.layer1 = self.make_layer(64, 64, num_blocks=2)
        self.layer2 = self.make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self.make_layer(128, 256, num_blocks=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 1)

    def make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), 1, -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
