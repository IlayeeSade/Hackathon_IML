import torch
import torch.nn as nn


class ModelArchitecture(nn.Module):
    """
    Student model architecture.

    Required behavior:
        input:  torch.Tensor of shape [batch_size, 3, height, width]
        output: torch.Tensor of shape [batch_size, 20]
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: batch of images

        Returns:
            logits for 20 classes
        """
        x = self.features(x)
        logits = self.classifier(x)
        return logits
