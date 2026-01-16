import torch
import torch.nn as nn

class CustomResidualBlock2(nn.Module):
    """
    Custom Residual Block consisting of three convolutional layers, one batch normalization layer, and an adaptation layer for the identity connection.

    Args:
        channels (int): Number of input and output channels.

    Returns:
        torch.Tensor: Output tensor after applying the residual block.
    """

    def __init__(self, channels: int):
        super(CustomResidualBlock2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(channels*2)
        self.relu3 = nn.ReLU()

        self.eyeAdapt = nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=1, padding=0, stride=1)

    
    def forward(self, x):
        
        # Adapt identity to match output channels
        identity = self.eyeAdapt(x)

        # First convolutional block
        out = self.conv1(x)
        out = self.relu(out)

        # Second convolutional block
        out = self.conv2(out)
        out = self.relu2(out)

        # Third convolutional block
        out = self.conv3(out)
        out = self.bn(out)
        out = self.relu3(out)

        # Add the input (identity) to the output
        out = out + identity
        out = self.relu(out)

        return out
    

class GC5(nn.Module):
    """
    Docstring for GC5 Model.
    """
    
    def __init__(self, num_classes: int):
        super(GC5, self).__init__()
        
        # --- Layer Definition ---
        # Output size: (image_size - kernel_size + 2*padding) / stride + 1
        # Receptive field: rl = rl−1 + (kl − 1) × jl−1
        # First convolutional layer (dim -> 256x256x32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1) # RF: 3
        self.relu = nn.ReLU()

        # First Residual Block
        self.resblock1 = CustomResidualBlock2(channels=32) # RF - conv1: 5 --> RF - conv2: 7 --> RF - conv3: 9

        # First pooling layer (dim -> 86x86x32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0) # RF: 11

        # Second Residual Block
        self.resblock2 = CustomResidualBlock2(channels=64) # RF - conv1: 17 --> RF - conv2: 23 --> RF - conv3: 29
        # Second pooling layer (dim -> 30x30x64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0) # RF: 35

        # Third Residual Block
        self.resblock3 = CustomResidualBlock2(channels=128) # RF - conv1: 53 --> RF - conv2: 71 --> RF - conv3: 89

        # Third pooling layer (dim -> 15x15x256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # RF: 98

        # Second convolutional layer (dim -> 8x8x256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2) # RF: 134
        self.relu4 = nn.ReLU()

        # Third convolutional layer (dim -> 3x3x256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=4, padding=0, stride=2) # RF: 242
        self.relu5 = nn.ReLU()


        # Global Average Pooling layer (dim -> 1x1x256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.4)

        # Output layer (dim -> num_classes)
        self.fc = nn.Linear(in_features=256, out_features=num_classes)

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
        out = self.relu(out)

        # Block 2
        out = self.resblock1(out)

        # First Pooling
        out = self.pool1(out)

        # Block 3
        out = self.resblock2(out)

        # Second Pooling
        out = self.pool2(out)

        # Block 4
        out = self.resblock3(out)

        # Third Pooling
        out = self.pool(out)

        # Block 5
        out = self.conv4(out)
        out = self.relu4(out)

        # Block 6
        out = self.conv5(out)
        out = self.relu5(out)

        # Global Average Pooling
        out = self.global_avg_pool(out) # dim -> 1x1x256
        out = torch.flatten(out, 1) # Flatten -> 256
        
        # Dropout
        out = self.dropout(out)

        # Fully Connected Layer
        out = self.fc(out)

        return out



        


