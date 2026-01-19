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
        
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=int(channels*2), kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=int(channels*2), out_channels=int(channels*3), kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()

        # Adaptation layer for identity connection
        self.eyeAdapt = nn.Conv2d(in_channels=channels, out_channels=int(channels*3), kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(int(channels*3))
        self.relu3 = nn.ReLU()
        
    

    def forward(self, x):
        
        # Adapt identity to match output channels
        identity = self.eyeAdapt(x)

        # First convolutional block
        out = self.conv1(x)
        out = self.relu(out)

        # Second convolutional block
        out = self.conv2(out)
        out = self.relu2(out)

        # Add the input (identity) to the output, apply ReLU and BatchNorm
        out = out + identity
        out = self.bn(out)
        out = self.relu3(out)

        return out
    

class GC5(nn.Module):
    """
    Final custom Convolutional Neural Network for Garbage Classification

    Layers:
        conv1: First convolutional layer (dim -> 256x256x4)
        pool1: First average pooling layer (dim -> 128x128x4)
        resblock1: First residual block (dim -> 128x128x12)
        pool2: Second average pooling layer (dim -> 64x64x12)
        resblock2: Second residual block (dim -> 64x64x36)
        pool3: Third average pooling layer (dim -> 32x32x36)
        resblock3: Third residual block (dim -> 32x32x108)
        pool4: Fourth average pooling layer (dim -> 16x16x108)
        resblock4: Fourth residual block (dim -> 16x16x324)
        pool5: Fifth average pooling layer (dim -> 8x8x324)
        conv2: Second convolutional layer (dim -> 4x4x512)
        global_avg_pool: Global average pooling layer (dim -> 1x1x512)
        dropout: Dropout layer
        fc: Output layer (dim -> num_classes)

    Notes:
        This model is fixed for RGB inputs only and expects tensors shaped as (N, 3, 256, 256).

    Args:
        num_classes (int): Number of output classes.
    """
    
    def __init__(self, num_classes: int):
        super(GC5, self).__init__()
        
        # --- Layer Definition ---
        # Output size: (image_size - kernel_size + 2*padding) / stride + 1
        # Receptive field: rl = rl−1 + (kl − 1) × jl−1
        # First convolutional layer (dim -> 256x256x4)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1, stride=1) # RF: 3
        self.relu = nn.ReLU()

        # First Pooling layer (dim -> 128x128x4)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # RF: 4 

        # First Residual Block (dim -> 128x128x12)
        self.resblock1 = CustomResidualBlock2(channels=4) # RF - conv1: 8  --> conv2: 12  --> conv3: 16
        
        # Second Pooling layer (dim -> 64x64x12)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # RF: 18

        # Second Residual Block (dim -> 64x64x36)
        self.resblock2 = CustomResidualBlock2(channels=12) # RF - conv1: 26  --> conv2: 34  --> conv3: 42

        # Third Pooling layer (dim -> 32x32x36)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2) # RF: 46

        # Third Residual Block (dim -> 32x32x108)
        self.resblock3 = CustomResidualBlock2(channels=36) # RF - conv1: 66 --> conv2: 82 --> conv3: 114

        # Fourth Pooling layer (dim -> 16x16x108)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2) # RF: 122

        # # Fourth Residual Block (dim -> 16x16x324)
        self.resblock4 = CustomResidualBlock2(channels=108) # RF - conv1: 138  --> conv2: 154  --> conv3: 170

        # Fifth Pooling layer (dim -> 8x8x324)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2) # RF: 186

        # Second convolutional layer (dim -> 4x4x512)
        self.conv2 = nn.Conv2d(324, 512, kernel_size=4, padding=1, stride=2) 
        self.relu2 = nn.ReLU()

        # Global Average Pooling layer (dim -> 1x1x512)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

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
        out = self.relu(out)

        # Pooling 1
        out = self.pool1(out)

        # Block 2
        out = self.resblock1(out)

        # Pooling 2
        out = self.pool2(out)

        # Block 3
        out = self.resblock2(out)

        # Pooling 3
        out = self.pool3(out)

        # Block 4
        out = self.resblock3(out)

        # Pooling 4
        out = self.pool4(out)

        # Block 5
        out = self.resblock4(out)

        # Pooling 5
        out = self.pool5(out)

        # Block 6
        out = self.conv2(out)
        out = self.relu2(out)

        # Global Average Pooling
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)  # Flatten

        # Dropout
        out = self.dropout(out)

        # Fully Connected Layer
        out = self.fc(out)

        return out



        


