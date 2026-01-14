import torch
import torch.nn as nn


class GC1(nn.Module):
    """
    First Custom Convolutional Neural Network for Garbage Classification

    Layers:
        conv1: First convolutional layer (dim -> 256x256x8)
        pool: First max pooling layer (dim -> 128x128x8)
        conv2: Second convolutional layer (dim -> 128x128x16)
        pool2: Second max pooling layer (dim -> 64x64x16)
        conv3: Third convolutional layer (dim -> 64x64x32)
        pool3: Third max pooling layer (dim -> 32x32x32)
        fc1: Fully connected layer (dim -> num_classes)
        softmax: Softmax layer (dim -> num_classes)

    Notes:
        This model is fixed for RGB inputs only and expects tensors shaped as (N, 3, 256, 256).

    Args:
        num_classes (int): Number of output classes.
    """
    def __init__(self, num_classes: int):
        super(GC1, self).__init__()
        
        # --- Layer Definition ---
        # Output size: (image_size - kernel_size + 2*padding) / stride + 1
        # Receptive field per layer: rl = rl−1 + (kl − 1) × jl−1
        # First convolutional layer (dim -> 256x256x8)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, stride=1) # RF: 3
        self.relu = nn.ReLU()

        # First max pooling layer (dim -> 128x128x8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        
        # Second convolutional layer (dim -> 128x128x16)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1) # RF: 5
        self.relu2 = nn.ReLU()

        # Second max pooling layer (dim -> 64x64x16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Third convolutional layer (dim -> 64x64x32)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1) # RF: 9
        self.relu3 = nn.ReLU()

        # Third max pooling layer (dim -> 32x32x32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layer (dim -> num_classes) 
        # Channels 32. Flatten -> 32 * 32 * 32
        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=num_classes)
        
        # Initializing weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights for convolutional and fully connected layers using Kaiming Normal initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"GC1 expects a 4D tensor (N, 3, 256, 256), got shape {tuple(x.shape)}")

        if x.size(1) != 3 or x.size(2) != 256 or x.size(3) != 256:
            raise ValueError(
                f"GC1 expects input shape (N, 3, 256, 256), got (N, {x.size(1)}, {x.size(2)}, {x.size(3)})"
            )

        # Block 1
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        
        # Block 2
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        
        # Block 3
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        
        # Flatten
        out = out.view(out.size(0), -1)
        
        # FC
        out = self.fc1(out)
        
        return out