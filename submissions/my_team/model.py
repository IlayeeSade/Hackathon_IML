import torch
import torch.nn as nn
import torchvision.models as models


class ModelArchitecture(nn.Module):
    """
    Student model architecture.

    Required behavior:
        input:  torch.Tensor of shape [batch_size, 3, height, width]
        output: torch.Tensor of shape [batch_size, 20]
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        self.resnet = models.resnet50(weights=None)
        
        in_features = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: batch of images

        Returns:
            logits for 20 classes
        """
        return self.resnet(x)
