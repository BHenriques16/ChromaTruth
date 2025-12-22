import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(1, 64) # Input: L Channel
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(256, 512)

        # Decoder + Skip Connections 
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256) # 256 (up) + 256 (skip)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128) # 128 (up) + 128 (skip)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64) # 64 (up) + 64 (skip)

        # Output Layer: ab channels
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.tanh = nn.Tanh() # Normalizes output to [-1, 1]

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        p1 = self.pool(s1)
        s2 = self.enc2(p1)
        p2 = self.pool(s2)
        s3 = self.enc3(p2)
        p3 = self.pool(s3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, s3], dim=1)) # Skip connection
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, s2], dim=1)) # Skip connection
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, s1], dim=1)) # Skip connection

        return self.tanh(self.final_conv(d1))