"""TCS-Fall official CNN class, retained unchanged; GPL-3.0 source attribution in README."""
import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.MaxPool2d(kernel_size=(5, 3)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.MaxPool2d(kernel_size=(5, 3)),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            # nn.MaxPool2d(kernel_size=(2,1)),
        )

        self.linear = nn.Linear(1280, 2)  # full connect layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # flatten on the dim 1
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x
