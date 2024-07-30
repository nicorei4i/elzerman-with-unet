import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
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
mask_name = 'sim_elzerman_traces_test_masks'  # Base name for the training HDF5 file
test_name = 'sim_elzerman_traces_test'  # Base name for the validation HDF5 file

# Construct full paths for the HDF5 files
hdf5_file_path_mask = os.path.join(current_dir, '{}.hdf5'.format(mask_name))  # Training file path
hdf5_file_path_test = os.path.join(current_dir, '{}.hdf5'.format(test_name))  # Validation file path

# Read data from the training HDF5 file
with h5py.File(hdf5_file_path_mask, 'r') as file:  # Open the HDF5 file in read mode
    all_keys = file.keys()  # Get all keys (datasets) in the HDF5 file
    data = np.array([file[key] for key in all_keys])  # Read data from each dataset and store in a numpy array
    print(data.shape)  # Print the data shape for verification

# Read data from the validation HDF5 file
with h5py.File(hdf5_file_path_test, 'r') as file:  # Open the HDF5 file in read mode
    all_keys = file.keys()  # Get all keys (datasets) in the HDF5 file
    mask_data = np.array([file[key] for key in all_keys])  # Read data from each dataset and store in a numpy array
    print(mask_data.shape)  # Print the validation data shape for verification

# Define parameters for the noise and simulation
noise_std = 0.3  # Standard deviation of Gaussian noise
T = 0.006  # Total simulation time in seconds
N = 2
n_samples = 10000
dt = T/(10000)
n_cycles = 2 #cycles per trace

# Define parameters for interference signals
# interference_amps = [1.1, 1.1, 1.1]  # Amplitudes of the interference signals
# interference_freqs = [50, 1000, 10 ** 4]  # Frequencies of the interference signals in Hz

interference_amps = [0.3, 0.3, 0.3, 0.3]  # Amplitudes of the interference signals
interference_freqs = [50, 200, 600, 1000]  # Frequencies of the interference signals in Hz

# Create an instance of the Noise class with the specified parameters
noise_transform = Noise(n_samples, T, noise_std, interference_amps, interference_freqs)

# Create an instance of the MinMaxScalerTransform class
test_scaler = MinMaxScalerTransform()
# Fit the scaler using data from the training HDF5 file
test_scaler.fit_from_hdf5(hdf5_file_path_test)
# Create instances of the SimDataset class for training and validation datasets
print('Creating datasets...')
test_dataset = SimDataset(hdf5_file_path_test, scale_transform=test_scaler, noise_transform=noise_transform)  # Training dataset
mask_dataset = SimDataset(hdf5_file_path_mask, scale_transform=None, noise_transform=None)  # Training dataset

# Create DataLoader
batch_size = 32  # Batch size for loading data
train_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # DataLoader for the dataset
val_loader = DataLoader(mask_dataset, batch_size=batch_size, shuffle=False)  # DataLoader for the validation dataset




current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
model_dir = os.path.join(current_dir, 'batch1k')
state_dict_name = 'model_weights'  # Base name for the model state dictionary
state_dict_path = os.path.join(model_dir, '{}.pth'.format(state_dict_name))  # Full path for saving model weights

model = UNet()
model.load_state_dict(torch.load(state_dict_path))
model.eval()


with torch.no_grad():  # Disable gradient calculation for validation
        for batch_x, batch_y in val_loader:  # Loop over each batch of validation data
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).squeeze(1).long()
            output = model(batch_x)  # Forward pass: get model output
            loss = criterion(output, batch_y)  # Calculate loss between output and target
            val_loss += loss.item() * batch_x.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)







# Visualize validation dataset predictions
x, y = next(iter(val_loader))  # Get a batch of validation data
x = x.to(device)
y = y.to(device)
model.eval()
with torch.no_grad():  # Disable gradient calculation for visualization
    decoded_test_data = model(x)
    m = torch.nn.Softmax(dim=1)
    decoded_test_data = m(decoded_test_data)
    decoded_test_data = decoded_test_data.cpu().numpy()  # Get model output for visualization
    predition_class = decoded_test_data.argmax(axis=1)
    
    x = x.cpu()
    y = y.cpu()
# Save model parameters

# Plot the results
print('Plotting the results...')
for i in range(10):
    fig, axs = plt.subplots(4, 1, figsize=(15, 5), sharex=True)  # Create a figure with 3 subplots

    fig.suptitle('Validation Traces')

    axs[0].plot(x[i].numpy().reshape(-1), label='Original', color='mediumblue', linewidth=0.9)
    axs[0].plot(predition_class[i], label='Original', color='orange', linewidth=0.9)

    axs[0].tick_params(labelbottom=False)
    axs[0].legend()
    #axs[0].set_xlim(0, 1000)

    axs[1].plot(decoded_test_data[i, 1, :], label = '$p(1)$', color = 'mediumblue', linewidth=0.9)
    axs[1].tick_params(labelbottom=False)
    axs[1].legend()
    

    
    plt.savefig(os.path.join(model_dir, f'validation_trace_{i}.png'))  # Save each figure
plt.show()


