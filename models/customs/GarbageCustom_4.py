import torch
import torch.nn as nn


class CustomResidualBlock(nn.Module):
    """
    Custom Residual Block with three convolutional layers and three batch normalization layers.

    Layers:
        conv1: First convolutional layer
        bn1: First batch normalization layer
        relu: ReLU activation
        conv2: Second convolutional layer
        bn2: Second batch normalization layer
        relu2: ReLU activation
        conv3: Third convolutional layer
        bn3: Third batch normalization layer
        relu3: ReLU activation
    
    Args:
        channels (int): Number of input and output channels.
    """

    def __init__(self, channels: int):
        super(CustomResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu3 = nn.ReLU()
    
    def forward(self, x):
        identity = x

        # First convolutional block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolutional block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # Third convolutional block
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        # Add the input (identity) to the output
        out = out + identity
        out = self.relu(out)

        return out
    

class GC4(nn.Module):
    """
    Fourth Custom Convolutional Neural Network for Garbage Classification

    Layers:
        conv1: First convolutional layer (dim -> 256x256x32)
        resblock1: First residual block
        conv2: Second convolutional layer (dim -> 86x86x64)
        resblock2: Second residual block
        conv3: Third convolutional layer (dim -> 30x30x128)
        resblock3: Third residual block
        pool: First max pooling layer (dim -> 15x15x128)
        conv4: Fourth convolutional layer (dim -> 8x8x256)
        conv5: Fifth convolutional layer (dim -> 8x8x512)
        global_avg_pool: Global average pooling layer (dim -> 1x1x512)
        dropout: Dropout layer
        fc: Output layer (dim -> num_classes)

    Notes:
        This model is fixed for RGB inputs only and expects tensors shaped as (N, 3, 256, 256).

    Args:
        num_classes (int): Number of output classes.
    """
    
    def __init__(self, num_classes: int):
        super(GC4, self).__init__()
        
        # --- Layer Definition ---
        # Output size: (image_size - kernel_size + 2*padding) / stride + 1
        # Receptive field: rl = rl−1 + (kl − 1) × jl−1
        # First convolutional layer (dim -> 256x256x32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False) # RF: 3
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        # First Residual Block
        self.resblock1 = CustomResidualBlock(channels=32) # RF - conv1: 5 --> RF - conv2: 7 --> RF - conv3: 9

        # Second convolutional layer (dim -> 86x86x64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=3, bias=False) # RF: 11
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        # Second Residual Block
        self.resblock2 = CustomResidualBlock(channels=64) # RF - conv1: 17 --> RF - conv2: 23 --> RF - conv3: 29

        # Third convolutional layer (dim -> 30x30x128)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=2, stride=3, bias=False) # RF: 35
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # Third Residual Block
        self.resblock3 = CustomResidualBlock(channels=128) # RF - conv1: 53 --> RF - conv2: 71 --> RF - conv3: 89

        # First max pooling layer (dim -> 15x15x128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # RF: 98

        # Fourth convolutional layer (dim -> 8x8x256)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False) # RF: 134
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        # Fifth convolutional layer (dim -> 8x8x512)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, bias=False) # RF: 206
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        # Global Average Pooling layer (dim -> 1x1x512)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.3)

        # Output layer (dim -> num_classes)
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

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
            raise ValueError(f"GC4 expects a 4D tensor (N, 3, 256, 256), got shape {tuple(x.shape)}")

        if x.size(1) != 3 or x.size(2) != 256 or x.size(3) != 256:
            raise ValueError(
                f"GC4 expects input shape (N, 3, 256, 256), got (N, {x.size(1)}, {x.size(2)}, {x.size(3)})"
            )

        # Block 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Block 2
        out = self.resblock1(out)

        # Block 3
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # Block 4
        out = self.resblock2(out)

        # Block 5
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        # Block 6
        out = self.resblock3(out)

        # Max Pooling
        out = self.pool(out)

        # Block 7
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        # Block 8
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)

        # Global Average Pooling
        out = self.global_avg_pool(out)  # Shape: (N, 512, 1, 1)
        out = out.view(out.size(0), -1)  # Flatten to (N, 512)

        # Dropout
        out = self.dropout(out)

        # Fully connected layer
        out = self.fc(out)

        return out



        


