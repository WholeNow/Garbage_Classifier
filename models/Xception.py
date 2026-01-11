import torch
import torch.nn as nn
import timm


class Xception(nn.Module):
    """
    Xception Model Wrapper using timm library.
    
    Args:
        num_classes (int): Number of output classes.
        model_name (str): Name of the Xception model variant in timm. Default is 'legacy_xception'.
        pretrained (bool): Whether to use pretrained weights. Default is True.
    """

    def __init__(self, num_classes: int, model_name: str = "legacy_xception", pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Xception expects a 4D tensor (N, 3, 299, 299), got shape {tuple(x.shape)}")

        if x.size(1) != 3 or x.size(2) != 299 or x.size(3) != 299:
            raise ValueError(
                f"Xception expects input shape (N, 3, 299, 299), got (N, {x.size(1)}, {x.size(2)}, {x.size(3)})"
            )
        return self.backbone(x)