import torch
import torch.nn as nn


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
        # Convolutional Block 1 with Residual Connection
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        self.residual_conv1 = nn.Conv1d(1, 32, kernel_size=1)  # Projection to match dimensions for residual

        # Self-Attention Module
        self.attention = SelfAttention1D(in_channels=32)

        # Convolutional Block 2 with Residual Connection
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.residual_conv2 = nn.Conv1d(32, 128, kernel_size=1)  # Projection to match dimensions for residual

        # MLP layers for complex feature mapping with Batch Normalization
        self.mlp = nn.Sequential(
            nn.Linear(128 * 1981, 512),  # Adjust size based on input length
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1981),  # Output back to original length
        )
        
        # Final convolution layer to map back to the original single channel
        self.output_layer = nn.Conv1d(1, 1, kernel_size=5, padding=2)

    def forward(self, x):
        # Convolutional Block 1 with Residual Connection
        residual1 = self.residual_conv1(x)
        x = self.conv_block1(x)
        x = x + residual1  # Adding residual connection
        x = self.attention(x)

        # Convolutional Block 2 with Residual Connection
        residual2 = self.residual_conv2(x)
        x = self.conv_block2(x)
        x = x + residual2  # Adding residual connection

        # Flatten for the MLP layers
        batch_size, channels, width = x.size()
        x = x.view(batch_size, -1)
        
        # Pass through MLP layers
        x = self.mlp(x)
        
        # Reshape back to (batch, channel, width) for final convolution
        x = x.view(batch_size, 1, width)
        x = self.output_layer(x)
        return x

class RamanNoiseNet_HF(nn.Module):
    def __init__(self):
        super(RamanNoiseNet_HF, self).__init__()
        # Convolutional Block 1 with Residual Connection
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        self.residual_conv1 = nn.Conv1d(1, 32, kernel_size=1)  # Projection to match dimensions for residual

        # Self-Attention Module
        self.attention = SelfAttention1D(in_channels=32)

        # Convolutional Block 2 with Residual Connection
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.residual_conv2 = nn.Conv1d(32, 128, kernel_size=1)  # Projection to match dimensions for residual

        # MLP layers for complex feature mapping with Batch Normalization
        self.mlp = nn.Sequential(
            nn.Linear(128 * 1900, 512),  # Adjust size based on input length
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1900),  # Output back to original length
        )
        
        # Final convolution layer to map back to the original single channel
        self.output_layer = nn.Conv1d(1, 1, kernel_size=5, padding=2)

    def forward(self, x):
        # Convolutional Block 1 with Residual Connection
        residual1 = self.residual_conv1(x)
        x = self.conv_block1(x)
        x = x + residual1  # Adding residual connection
        x = self.attention(x)

        # Convolutional Block 2 with Residual Connection
        residual2 = self.residual_conv2(x)
        x = self.conv_block2(x)
        x = x + residual2  # Adding residual connection

        # Flatten for the MLP layers
        batch_size, channels, width = x.size()
        x = x.view(batch_size, -1)
        
        # Pass through MLP layers
        x = self.mlp(x)
        
        # Reshape back to (batch, channel, width) for final convolution
        x = x.view(batch_size, 1, width)
        x = self.output_layer(x)
        return x

class RamanNoiseNet_LF(nn.Module):
    def __init__(self):
        super(RamanNoiseNet_LF, self).__init__()
        # Convolutional Block 1 with Residual Connection
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        self.residual_conv1 = nn.Conv1d(1, 32, kernel_size=1)  # Projection to match dimensions for residual

        # Self-Attention Module
        self.attention = SelfAttention1D(in_channels=32)

        # Convolutional Block 2 with Residual Connection
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.residual_conv2 = nn.Conv1d(32, 128, kernel_size=1)  # Projection to match dimensions for residual

        # MLP layers for complex feature mapping with Batch Normalization
        self.mlp = nn.Sequential(
            nn.Linear(128 * 81, 512),  # Adjust size based on input length
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 81),  # Output back to original length
        )
        
        # Final convolution layer to map back to the original single channel
        self.output_layer = nn.Conv1d(1, 1, kernel_size=5, padding=2)

    def forward(self, x):
        # Convolutional Block 1 with Residual Connection
        residual1 = self.residual_conv1(x)
        x = self.conv_block1(x)
        x = x + residual1  # Adding residual connection
        x = self.attention(x)

        # Convolutional Block 2 with Residual Connection
        residual2 = self.residual_conv2(x)
        x = self.conv_block2(x)
        x = x + residual2  # Adding residual connection

        # Flatten for the MLP layers
        batch_size, channels, width = x.size()
        x = x.view(batch_size, -1)
        
        # Pass through MLP layers
        x = self.mlp(x)
        
        # Reshape back to (batch, channel, width) for final convolution
        x = x.view(batch_size, 1, width)
        x = self.output_layer(x)
        return x

class RamanNoiseNet_Clip(nn.Module):
    def forward(self, x):
        x_low = x[:,:,:81]
        x_high = x[:,:,81:]
        model_LF = RamanNoiseNet_LF()
        model_HF = RamanNoiseNet_HF()
        x_low = model_LF(x_low)
        x_high = model_HF(x_high)
        x = torch.cat((x_low, x_high), dim=2)
        return x


if __name__ == "__main__":
    length = 1900  # Example spectrum length
    num = 100 # number of spectrum
    random_input = torch.randn(num, 1, length)  # Shape: (batch_size, channels, length)

    # Initialize the model
    model = RamanNoiseNet_HF()

    # Forward pass
    output = model(random_input)

    # Print output shape to verify it works
    print(f"Input shape: {random_input.shape}")
    print(f"Output shape: {output.shape}")