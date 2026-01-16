import torch
import torch.nn as nn


class GC3(nn.Module):
    """
    Third Custom Convolutional Neural Network for Garbage Classification

    Layers:
        conv1: First convolutional layer (dim -> 256x256x16)
        conv2: Second convolutional layer (dim -> 128x128x32)
        conv3: Third convolutional layer (dim -> 64x64x64)
        pool: First max pooling layer (dim -> 32x32x64)
        conv4: Fourth convolutional layer (dim -> 16x16x128)
        conv5: Fifth convolutional layer (dim -> 8x8x128)
        conv6: Sixth convolutional layer (dim -> 4x4x256)
        fc1: Fully connected layer (dim -> 1024)
        dropout: Dropout layer
        fc2: Output layer (dim -> num_classes)

    Notes:
        This model is fixed for RGB inputs only and expects tensors shaped as (N, 3, 256, 256).

    Args:
        num_classes (int): Number of output classes.
    """
    
    def __init__(self, num_classes: int):
        super(GC3, self).__init__()
        # --- Layer Definition ---
        # Output size: (image_size - kernel_size + 2*padding) / stride + 1
        # Receptive field: rl = rl−1 + (kl − 1) × jl−1
        # First convolutional layer (dim -> 256x256x16)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False) # RF: 3
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # Second convolutional layer (dim -> 128x128x32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False) # RF: 5
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        # Third convolutional layer (dim -> 64x64x64)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False) # RF: 9
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        # First max pooling layer (dim -> 32x32x64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # RF: 17 

        # Fourth convolutional layer (dim -> 16x16x128)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False) # RF: 33
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()

        # Fifth convolutional layer (dim -> 8x8x128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=False) # RF: 65
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()

        # Sixth convolutional layer (dim -> 4x4x256)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False) # RF: 129
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()

        # Fully connected layer (dim -> num_classes)
        # Channels 256. Flatten -> 256 * 4 * 4
        self.fc1 = nn.Linear(in_features=256 * 4 * 4, out_features=1024)
        self.relu_fc1 = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

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
            raise ValueError(f"GC3 expects a 4D tensor (N, 3, 256, 256), got shape {tuple(x.shape)}")

        if x.size(1) != 3 or x.size(2) != 256 or x.size(3) != 256:
            raise ValueError(
                f"GC3 expects input shape (N, 3, 256, 256), got (N, {x.size(1)}, {x.size(2)}, {x.size(3)})"
            )

        # Block 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Block 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # Block 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out = self.pool(out)

        # Block 4
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        # Block 5
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)

        # Block 6
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu6(out)

        # Flatten
        out = out.view(out.size(0), -1)

        # FC layers
        out = self.fc1(out)
        out = self.relu_fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out