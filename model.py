import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu

class encoder_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder_block, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)        

        x = self.conv2(x)
        x = self.relu(x)

        xp = self.pool(x)
        return xp, x

class decoder_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_block, self).__init__()
        self.upconv = torch.nn.ConvTranspose1d(in_channels, out_channels, 2, stride=2)
        
        self.conv1 = torch.nn.Conv1d(in_channels + out_channels, out_channels, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.conv3 = torch.nn.Conv1d(out_channels, out_channels, 3, padding=1)
        

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(4)

    def forward(self, x, encoder_x):
        x = self.upconv(x)
        
        diff = encoder_x.size(2) - x.size(2)
        if diff > 0:
            x = nn.functional.pad(x, (diff // 2, diff - diff // 2))
        
        x = self.conv1(torch.cat([x, encoder_x], dim=1))
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)    
        x = self.conv3(x)
        x = self.relu(x)       


        return x


class bottleneck_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(bottleneck_block, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.conv3 = torch.nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = encoder_block(1, 4)
        self.encoder2 = encoder_block(4, 8)
        self.encoder3 = encoder_block(8, 32)
        
        self.bottleneck = bottleneck_block(32, 32)
        
        self.decoder1 = decoder_block(32, 8)
        self.decoder2 = decoder_block(8, 4)
        self.decoder4 = decoder_block(4, 2)
        
        
    def forward(self, x):
        xp1,x1 = self.encoder1(x)
        xp2,x2 = self.encoder2(xp1)
        xp3,x3 = self.encoder3(xp2)
        
        xb = self.bottleneck(xp3)
        
        x4 = self.decoder1(xb, x3)
        x5 = self.decoder2(x4, x2)
        x6 = self.decoder4(x5, x1)
        return x6