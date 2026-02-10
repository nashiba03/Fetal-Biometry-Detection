# UNet.py 
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)  
        self.enc4 = DoubleConv(128, 256)  
        self.pool = nn.MaxPool2d(2)

        
        self.bottleneck = DoubleConv(256, 512)

        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = DoubleConv(512, 256)  

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)  

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)   

        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec4 = DoubleConv(64, 32)    

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        
        b = self.bottleneck(self.pool(e4))

        
        d1 = self.up1(b)
        d1 = F.interpolate(d1, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e4], dim=1))

        d2 = self.up2(d1)
        d2 = F.interpolate(d2, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e3], dim=1))

        d3 = self.up3(d2)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d4 = self.up4(d3)
        d4 = F.interpolate(d4, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e1], dim=1))

        return torch.sigmoid(self.out(d4))