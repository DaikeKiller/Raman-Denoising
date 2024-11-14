import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention) * x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class AUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AUnet, self).__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.attn1 = nn.Sequential(SpatialAttention(), ChannelAttention(64))
        self.enc2 = DoubleConv(64, 128)
        self.attn2 = nn.Sequential(SpatialAttention(), ChannelAttention(128))
        self.enc3 = DoubleConv(128, 256)
        self.attn3 = nn.Sequential(SpatialAttention(), ChannelAttention(256))
        self.enc4 = DoubleConv(256, 512)
        self.attn4 = nn.Sequential(SpatialAttention(), ChannelAttention(512))
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.attn_dec4 = nn.Sequential(SpatialAttention(), ChannelAttention(512))
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.attn_dec3 = nn.Sequential(SpatialAttention(), ChannelAttention(256))
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.attn_dec2 = nn.Sequential(SpatialAttention(), ChannelAttention(128))
        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.attn_dec1 = nn.Sequential(SpatialAttention(), ChannelAttention(64))
        
        # Output layer
        self.out_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc1 = self.attn1(enc1)
        enc2 = self.enc2(F.max_pool1d(enc1, kernel_size=2, ceil_mode=True))
        enc2 = self.attn2(enc2)
        enc3 = self.enc3(F.max_pool1d(enc2, kernel_size=2, ceil_mode=True))
        enc3 = self.attn3(enc3)
        enc4 = self.enc4(F.max_pool1d(enc3, kernel_size=2, ceil_mode=True))
        enc4 = self.attn4(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool1d(enc4, kernel_size=2, ceil_mode=True))
        
        # Decoder path
        dec4 = self.upconv4(bottleneck)
        dec4 = F.interpolate(dec4, size=enc4.size(2))  # Ensure matching size for concatenation
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)
        dec4 = self.attn_dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = F.interpolate(dec3, size=enc3.size(2))  # Ensure matching size for concatenation
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)
        dec3 = self.attn_dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = F.interpolate(dec2, size=enc2.size(2))  # Ensure matching size for concatenation
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)
        dec2 = self.attn_dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = F.interpolate(dec1, size=enc1.size(2))  # Ensure matching size for concatenation
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)
        dec1 = self.attn_dec1(dec1)
        
        # Output layer
        out = self.out_conv(dec1)
        return out

# Example usage
if __name__ == "__main__":
    model = AUnet(in_channels=1, out_channels=1)
    x = torch.randn(8, 1, 81)  # Batch size of 8, 1 channel, length 81
    out = model(x)
    print(out.shape)  # Should be (8, 1, 81)
