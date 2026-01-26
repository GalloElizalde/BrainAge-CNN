import torch
import torch.nn as nn

class BrainAgeCNN(nn.Module):
    def __init__(self, in_channels: int = 15):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(64,1),
        )

    def forward(self,x):
        x = self.sequence(x)
        x = x.view(x.size(0), -1)     # (B, 64)
        x = self.head(x)              # (B, 1)
        return x.squeeze(1)           # (B,)
    
