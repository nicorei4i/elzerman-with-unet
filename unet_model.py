import torch
import torch.nn as nn

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the pooling size
pool_size = 8

# Define the encoder block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        # Two convolutional layers with ReLU activation
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        # Max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, ceil_mode=True)

    def forward(self, x):
        # Apply first convolution and ReLU
        x = self.conv1(x)
        x = self.relu(x)
        # Apply second convolution and ReLU
        x = self.conv2(x)
        x = self.relu(x)
        # Apply max pooling and return both pooled and original feature maps
        xp = self.pool(x)
        return xp, x

# Define the bottleneck block
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckBlock, self).__init__()
        # Two convolutional layers with ReLU activation
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.GELU()

    def forward(self, x):
        # Apply first convolution and ReLU
        x = self.conv1(x)
        x = self.relu(x)
        # Apply second convolution and ReLU
        x = self.conv2(x)
        x = self.relu(x)
        return x

# Define the decoder block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # Transposed convolutional layer for upsampling
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, pool_size, stride=pool_size, padding=0)
        # Two convolutional layers with ReLU activation
        self.conv1 = nn.Conv1d(in_channels + out_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.GELU()

    def forward(self, x, encoder_x):
        # Apply transposed convolution for upsampling
        x = self.upconv(x)
        # Crop and concatenate the encoder feature map with the upsampled feature map
        x = self._crop_and_concat(encoder_x, x)
        # Apply first convolution and ReLU
        x = self.conv1(x)
        x = self.relu(x)
        # Apply second convolution and ReLU
        x = self.conv2(x)
        x = self.relu(x)
        return x

    def _crop_and_concat(self, enc_features, x):
        # Get the lengths of the feature maps
        _, _, x_len = x.size()
        enc_len = enc_features.size(2)
        # Crop or pad the encoder features to match the size of the upsampled features
        if enc_len > x_len:
            enc_features = enc_features[:, :, :x_len]
        elif enc_len < x_len:
            pad = x_len - enc_len
            enc_features = nn.functional.pad(enc_features, (0, pad))
        # Concatenate the encoder features and the upsampled features
        return torch.cat((enc_features, x), dim=1)

# Define the UNet model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define the encoder blocks
        self.encoder1 = EncoderBlock(1, 4).to(device)
        self.encoder2 = EncoderBlock(4, 16).to(device)
        self.encoder3 = EncoderBlock(16, 32).to(device)
        # Define the bottleneck block
        self.bottleneck = BottleneckBlock(32, 32).to(device)
        # Define the decoder blocks
        self.decoder1 = DecoderBlock(32, 16).to(device)
        self.decoder2 = DecoderBlock(16, 4).to(device)
        self.decoder3 = DecoderBlock(4, 2).to(device)
        # Define a final convolutional layer to ensure output matches the input size
        

    def forward(self, x):
        # Pass input through encoder blocks
        xp1, x1 = self.encoder1(x)
        xp2, x2 = self.encoder2(xp1)
        xp3, x3 = self.encoder3(xp2)
        # Pass through bottleneck block
        xb = self.bottleneck(xp3)
        # Pass through decoder blocks
        x4 = self.decoder1(xb, x3)
        x5 = self.decoder2(x4, x2)
        x6 = self.decoder3(x5, x1)
        # Apply final convolution to adjust the number of channels
        # Crop the output to match the input size
        x6 = x6[:, :, :x.size(2)]
        return x6
