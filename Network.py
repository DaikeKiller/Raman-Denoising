import torch
import torch.nn as nn
import numpy as np


# Self-Attention Module
class SelfAttention1D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention1D, self).__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, width)
        key = self.key_conv(x).view(batch_size, -1, width)
        energy = torch.bmm(query.permute(0, 2, 1), key)
        attention = self.softmax(energy)
        value = self.value_conv(x).view(batch_size, -1, width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = self.gamma * out.view(batch_size, channels, width) + x
        return out

# Modified Neural Network with Self-Attention
class RamanNoiseNet(nn.Module):
    def __init__(self):
        super(RamanNoiseNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.attention = SelfAttention1D(in_channels=32)
        self.output_layer = nn.Conv1d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.conv_layers(x)
        print(x.shape)
        x = self.attention(x)
        print(x.shape)
        x = self.output_layer(x)
        print(x.shape)
        return x

if __name__ == "__main__":
    length = 1000  # Example spectrum length
    num = 100 # number of spectrum
    random_input = torch.randn(num, 1, length)  # Shape: (batch_size, channels, length)

    # Initialize the model
    model = RamanNoiseNet()

    # Forward pass
    output = model(random_input)

    # Print output shape to verify it works
    print(f"Input shape: {random_input.shape}")
    print(f"Output shape: {output.shape}")