import torch.nn as nn


class Conv1DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv1DAutoencoder, self).__init__()

        # Define the encoder part of the autoencoder
        self.encoder = nn.Sequential(
            # First convolutional layer: 1 input channel, 16 output channels, kernel size 4, stride 2, padding 1
            nn.Conv1d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # Activation function

            # Second convolutional layer: 16 input channels, 32 output channels, kernel size 4, stride 2, padding 1
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # Activation function

            # Third convolutional layer: 32 input channels, 64 output channels, kernel size 4, stride 2, padding 1
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()  # Activation function
        )

        # Define the decoder part of the autoencoder
        self.decoder = nn.Sequential(
            # First transposed convolutional layer: 64 input channels, 32 output channels, kernel size 4, stride 2, padding 1
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # Activation function

            # Second transposed convolutional layer: 32 input channels, 16 output channels, kernel size 4, stride 2, padding 1
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # Activation function

            # Third transposed convolutional layer: 16 input channels, 1 output channel, kernel size 4, stride 2, padding 1
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid activation function to ensure output values are between 0 and 1
        )

    def forward(self, x):
        """
        Define the forward pass of the autoencoder.

        Parameters:
        x (Tensor): Input tensor of shape (batch_size, 1, sequence_length)

        Returns:
        Tensor: Reconstructed tensor of the same shape as input
        """
        encoded = self.encoder(x)  # Pass the input through the encoder
        decoded = self.decoder(encoded)  # Pass the encoded representation through the decoder
        return decoded  # Return the reconstructed output

