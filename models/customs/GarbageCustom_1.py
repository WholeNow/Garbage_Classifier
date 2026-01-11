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

    Args:
        input_size (int): Number of input channels
    """
    def __init__(self, input_size, num_classes):
        super(GC1, self).__init__()
        
        # --- Layer Definition ---
        # First convolutional layer (dim -> 256x256x8)
        self.conv1 = nn.Conv2d(input_size, 8, kernel_size=3, padding=1, stride=1)  
        self.relu = nn.ReLU()

        # First max pooling layer (dim -> 128x128x8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        
        # Second convolutional layer (dim -> 128x128x16)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()

        # Second max pooling layer (dim -> 64x64x16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Third convolutional layer (dim -> 64x64x32)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU()

        # Third max pooling layer (dim -> 32x32x32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layer (dim -> num_classes)
        # Input image 256 -> pool 128 -> pool 64 -> pool 32. 
        # Channels 32. Flatten -> 32 * 32 * 32
        self.fc1 = nn.Linear(32 * 32 * 32, num_classes)
        
        # Initializing weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Inizializza i pesi con He Initialization (Kaiming Normal).
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

        return out