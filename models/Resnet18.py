import torch
import torch.nn as nn
import torchvision.models as models

class Resnet18(nn.Module):
    """
    Resnet18 Model Wrapper.
    
    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use pretrained weights. Default is True.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        # torchvision>=0.13 deprecates `pretrained=` in favor of `weights=`.
        try:
            from torchvision.models import ResNet18_Weights

            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            # Fallback for older torchvision versions
            self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Resnet18 expects a 4D tensor (N, 3, 224, 224), got shape {tuple(x.shape)}")

        if x.size(1) != 3 or x.size(2) != 224 or x.size(3) != 224:
            raise ValueError(
                f"Resnet18 expects input shape (N, 3, 224, 224), got (N, {x.size(1)}, {x.size(2)}, {x.size(3)})"
            )
        return self.backbone(x)