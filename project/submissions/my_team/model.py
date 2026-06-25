import torch
import torch.nn as nn
import torchvision.models as models

class ModelArchitecture(nn.Module):
    """
    Student model architecture using ConvNeXt-Tiny trained from scratch.
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        # Loading the ConvNeXt-Tiny architecture with randomly initialized weights
        self.convnext = models.convnext_tiny(weights=None)

        # ConvNeXt classifier head is located in self.convnext.classifier
        # The last linear layer inside it is at index [2]
        in_features: int = self.convnext.classifier[2].in_features

        # Rebuilding the classifier head with Dropout for improved robustness
        self.convnext.classifier = nn.Sequential(
            self.convnext.classifier[0],  # LayerNorm from the original architecture
            self.convnext.classifier[1],  # Flatten from the original architecture (B, C, 1, 1) -> (B, C)
            nn.Dropout(p=0.3),            # Regularization layer
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        return self.convnext(x)