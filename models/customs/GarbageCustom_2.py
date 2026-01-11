import torch
import torch.nn as nn


class GC2(nn.Module):


    def __init__(self, num_classes:int):
        super(GC2, self).__init__()

        # --- Layer Definition ---
        # Output size: (image_size - kernel_size + 2*padding) / stride + 1
        # First convolutional layer (dim -> 256x256x12)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()

        # Second convolutional layer (dim -> 126x126x16)
        self.conv2 = nn.Conv2d(12, 16, kernel_size=5, padding=1, stride=2)
        self.relu2 = nn.ReLU()

        # First max pooling layer (dim -> 63x63x16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Third convolutional layer (dim -> 63x63x24)
        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU()

        # Fourth convolutional layer (dim -> 30x30x32)
        self.conv4 = nn.Conv2d(24, 32, kernel_size=5, padding=1, stride=2)
        self.relu4 = nn.ReLU()

        # Second max pooling layer (dim -> 15x15x32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layer (dim -> 1024)
        # Input image 256 -> conv2 stride 2 -> 128 -> pool 64 -> conv4 stride 2 -> 30 -> pool2 15. 
        # Channels 32. Flatten -> 32 * 15 * 15
        self.fc1 = nn.Linear(in_features=32 * 15 * 15, out_features= 1024)
        self.relu_fc1 = nn.ReLU()
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.2)

        # Output layer (dim -> num_classes)
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
            raise ValueError(f"GC1 expects a 4D tensor (N, 3, 256, 256), got shape {tuple(x.shape)}")

        if x.size(1) != 3 or x.size(2) != 256 or x.size(3) != 256:
            raise ValueError(
                f"GC1 expects input shape (N, 3, 256, 256), got (N, {x.size(1)}, {x.size(2)}, {x.size(3)})"
            )

        # Block 1
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool(out)

        # Block 2
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.pool2(out)

        # Flatten
        out = out.view(out.size(0), -1)

        # FC layers
        out = self.fc1(out)
        out = self.relu_fc1(out)
        out = self.fc2(out)

        return out


