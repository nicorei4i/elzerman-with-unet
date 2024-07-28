#%%

import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import UNet
from torch.utils.data import DataLoader
from dataset import SimDataset, Noise, MinMaxScalerTransform
from sklearn.preprocessing import MinMaxScaler
print('GPU available: ', torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Directory setup
current_dir = os.getcwd()  # Get the current working directory
file_name = 'sim_elzerman_traces_train'  # Base name for the training HDF5 file
val_name = 'sim_elzerman_traces_val'  # Base name for the validation HDF5 file

# Construct full paths for the HDF5 files
hdf5_file_path = os.path.join(current_dir, '{}.hdf5'.format(file_name))  # Training file path
hdf5_file_path_val = os.path.join(current_dir, '{}.hdf5'.format(val_name))  # Validation file path

# Read data from the training HDF5 file
with h5py.File(hdf5_file_path, 'r') as file:  # Open the HDF5 file in read mode
    all_keys = file.keys()  # Get all keys (datasets) in the HDF5 file
    data = np.array([file[key] for key in all_keys])  # Read data from each dataset and store in a numpy array
    print(data.shape)  # Print the data shape for verification

# Read data from the validation HDF5 file
with h5py.File(hdf5_file_path_val, 'r') as file:  # Open the HDF5 file in read mode
    all_keys = file.keys()  # Get all keys (datasets) in the HDF5 file
    val_data = np.array([file[key] for key in all_keys])  # Read data from each dataset and store in a numpy array
    print(val_data.shape)  # Print the validation data shape for verification

# Define parameters for the noise and simulation
noise_std = 0.1  # Standard deviation of Gaussian noise
T = 0.006  # Total simulation time in seconds
N = 2
n_samples = 10000
dt = T/(10000)
n_cycles = 2 #cycles per trace

# Define parameters for interference signals
# interference_amps = [1.1, 1.1, 1.1]  # Amplitudes of the interference signals
# interference_freqs = [50, 1000, 10 ** 4]  # Frequencies of the interference signals in Hz

interference_amps = [0.1, 0.1, 0.1, 0.1]  # Amplitudes of the interference signals
interference_freqs = [50, 200, 600, 1000]  # Frequencies of the interference signals in Hz


# Create an instance of the Noise class with the specified parameters
noise_transform = Noise(n_samples, T, noise_std, interference_amps, interference_freqs)

# Create an instance of the MinMaxScalerTransform class
train_scaler = MinMaxScalerTransform()
val_scaler = MinMaxScalerTransform()
# Fit the scaler using data from the training HDF5 file
train_scaler.fit_from_hdf5(hdf5_file_path)
val_scaler.fit_from_hdf5(hdf5_file_path_val)
# Create instances of the SimDataset class for training and validation datasets
print('Creating datasets...')
dataset = SimDataset(hdf5_file_path, scale_transform=train_scaler, noise_transform=noise_transform)  # Training dataset
val_dataset = SimDataset(hdf5_file_path_val, scale_transform=val_scaler, noise_transform=noise_transform)  # Validation dataset

# Create DataLoader
batch_size = 32  # Batch size for loading data
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # DataLoader for the dataset
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)  # DataLoader for the validation dataset

# Initialize model, loss function, and optimizer
model = UNet(1).to(device)  # Instance of the Conv1DAutoencoder model
#criterion = nn.CrossEntropyLoss()  # Cross Entropy loss function
criterion = nn.MSELoss().to(device)  # Mean Squared Error loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer with learning rate 0.001

# Training loop with validation
print('Start training...')
num_epochs = 25  # Number of epochs for training, adjust based on the final loss
for epoch in range(num_epochs):  # Loop over each epoch
    # Training loop
    print('...still training...')
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:  # Loop over each batch of data
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        output = model(batch_x)  # Forward pass: get model output
        loss = criterion(output, batch_y)  # Calculate loss between output and target

        optimizer.zero_grad()  # Zero the gradients before backward pass
        loss.backward()  # Backward pass: compute gradient of the loss
        optimizer.step()  # Update model parameters

        train_loss += loss.item() * batch_x.size(0)
    train_loss /= len(train_loader.dataset)

    # Print the loss for each epoch
    #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Validation loop
    print('...validating...')
    model.eval()
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch_x, batch_y in val_loader:  # Loop over each batch of validation data
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x)  # Forward pass: get model output
            loss = criterion(output, batch_y)  # Calculate loss between output and target
            val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

    # Print the loss for each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')


#%%
# Visualize validation dataset predictions
x, y = next(iter(val_loader))  # Get a batch of validation data
x = x.to(device)
y = y.to(device)
model.eval()
with torch.no_grad():  # Disable gradient calculation for visualization
    decoded_test_data = model(x)
    decoded_test_data = decoded_test_data.cpu().numpy()  # Get model output for visualization
    x = x.cpu()
    y = y.cpu()
# Save model parameters
print('Saving model parameters...')
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
model_dir = os.path.join(current_dir, 'batch32_lr01')
os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist
state_dict_name = 'model_weights'  # Base name for the model state dictionary
state_dict_path = os.path.join(model_dir, '{}.pth'.format(state_dict_name))  # Full path for saving model weights
torch.save(model.state_dict(), state_dict_path)  # Save the model state dictionary to file

# Plot the results
print('Plotting the results...')
for i in range(10):
    fig, axs = plt.subplots(3, 1, figsize=(15, 5))  # Create a figure with 3 subplots

    fig.suptitle('Validation Traces')

    axs[0].plot(x[i].numpy().reshape(-1), label='Noisy', color='mediumblue', linewidth=0.9)
    axs[0].tick_params(labelbottom=False)
    axs[0].legend()
    #axs[0].set_xlim(0, 1000)

    axs[1].plot(decoded_test_data[i].reshape(-1), label='Denoised', color='mediumblue', linewidth=0.9)
    axs[1].tick_params(labelbottom=False)
    axs[1].legend()
    #axs[1].set_xlim(0, 1000)

    axs[2].plot(y[i].numpy().reshape(-1), label='Clean', color='mediumblue', linewidth=0.9)
    axs[2].legend()
    #axs[2].set_xlim(0, 1000)

    plt.savefig(os.path.join(model_dir, f'validation_trace_{i}.png'))  # Save each figure
    plt.show()

