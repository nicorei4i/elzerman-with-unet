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

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directory setup
current_dir = os.getcwd()  # Get the current working directory
file_name = 'sim_elzerman_traces_test_10k'  # Name for the test HDF5 file
#mask_name = 'sim_elzerman_test_masks'  # Name for the mask HDF5 file

# Construct full paths for the HDF5 files
hdf5_file_path = os.path.join(current_dir, '{}.hdf5'.format(file_name))  # Test file path
#hdf5_file_path_masks = os.path.join(current_dir, '{}.hdf5'.format(mask_name))  # Mask file path

# Read data from the test HDF5 file
with h5py.File(hdf5_file_path, 'r') as file:
    all_keys = file.keys()  # Get all keys (datasets) in the HDF5 file
    test_data = np.array([file[key] for key in all_keys])  # Read data and store in a numpy array
    print(test_data.shape)  # Print the data shape for verification

# Read data from the mask HDF5 file
""" with h5py.File(hdf5_file_path_masks, 'r') as file:
    all_keys = file.keys()  # Get all keys (datasets) in the HDF5 file
    mask_data = np.array([file[key] for key in all_keys])  # Read data and store in a numpy array
    print(mask_data.shape)  # Print the data shape for verification
 """
# Define time indices
t_L, t_W, t_R, t_U = 0.5e-3, 0.0, 1.0e-3, 1.5e-3  # Time intervals in seconds
times = np.array([t_L, t_W, t_R, t_U])
trace_duration = np.sum(times)
times = np.array([t_L, t_L + t_W, t_L + t_W + t_R, t_L + t_W + t_R + t_U])
n_samples = 8192
times_indices = times * n_samples / trace_duration
times_indices = times_indices.astype(np.int64)
start_read, end_read = times_indices[1], times_indices[2]

# Define parameters for the noise and simulation
noise_std = 0.3  # Standard deviation of Gaussian noise
T = trace_duration  # Total simulation time in seconds
N = 2
n_samples = 8192
dt = T / 8192
n_cycles = 2  # Cycles per trace

# Define parameters for interference signals
interference_amps = [0.3, 0.3, 0.3, 0.3]  # Amplitudes of the interference signals
interference_freqs = [50, 200, 600, 1000]  # Frequencies of the interference signals in Hz

# Create an instance of the Noise class
noise_transform = Noise(n_samples, T, noise_std, interference_amps, interference_freqs)

# Create an instance of the MinMaxScalerTransform class
scaler = MinMaxScalerTransform()
scaler.fit_from_hdf5(hdf5_file_path)  # Fit the scaler using data from the test HDF5 file

# Create instances of the SimDataset class for validation datasets
print('Creating datasets...')
val_dataset = SimDataset(hdf5_file_path, scale_transform=scaler, noise_transform=noise_transform)  # Validation dataset
#mask_dataset = SimDataset(hdf5_file_path_masks, scale_transform=None, noise_transform=None)  # Mask dataset

# Create DataLoader
batch_size = 32  # Batch size for loading data
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # DataLoader for validation dataset
#mask_loader = DataLoader(mask_dataset, batch_size=batch_size, shuffle=False)  # DataLoader for mask dataset

# Model setup
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
model_dir = os.path.join(current_dir, 'batch10k')  # Directory for model weights
state_dict_name = 'model_weights'  # Name for the model state dictionary
state_dict_path = os.path.join(model_dir, '{}.pth'.format(state_dict_name))  # Full path for saving model weights

# Load the model
model = UNet()
model.load_state_dict(torch.load(state_dict_path))
model.eval()

# Visualize validation dataset predictions
x, y = next(iter(test_loader))  # Get a batch of validation data
x = x.to(device)
y = y.to(device)
model.eval()
with torch.no_grad():  # Disable gradient calculation for visualization
    decoded_test_data = model(x)
    m = torch.nn.Softmax(dim=1)
    decoded_test_data = m(decoded_test_data)
    decoded_test_data = decoded_test_data.cpu().numpy()  # Get model output for visualization
    prediction_class = decoded_test_data.argmax(axis=1)
    
    x = x.cpu()
    y = y.cpu()

# Plot the results
print('Plotting the results...')
for i in range(10):
    fig, axs = plt.subplots(4, 1, figsize=(15, 5), sharex=True)  # Create a figure with 4 subplots

    fig.suptitle('Validation Traces')

    axs[0].plot(x[i].numpy().reshape(-1), label='Noisy', color='mediumblue', linewidth=0.9)
    axs[0].tick_params(labelbottom=False)
    axs[0].legend()

    axs[1].plot(prediction_class[i], label='Denoised', color='mediumblue', linewidth=0.9)
    axs[1].tick_params(labelbottom=False)
    axs[1].legend()

    axs[2].plot(decoded_test_data[i, 1, :], label='$p(1)$', color='mediumblue', linewidth=0.9)
    axs[2].tick_params(labelbottom=False)
    axs[2].legend()

    axs[3].plot(y[i].numpy().reshape(-1), label='Clean', color='mediumblue', linewidth=0.9)
    axs[3].legend()

    plt.savefig(os.path.join(model_dir, f'validation_trace_{i}.png'))  # Save each figure

# Initialize confusion matrix components
nfn, nfp, ntn, ntp = 0, 0, 0, 0

def invert(arr):
    where_0 = np.where(arr == 0)
    where_1 = np.where(arr == 1)
    arr[where_0] = 1
    arr[where_1] = 0
    return arr

# Validate the model
with torch.no_grad():  # Disable gradient calculation for validation
    for batch_x, batch_y in test_loader:  # Loop over each batch of validation data
        decoded_test_data = model(batch_x)
        batch_y = batch_y.numpy()
        batch_y = batch_y.squeeze(1)
        m = torch.nn.Softmax(dim=1)
        decoded_test_data = m(decoded_test_data)
        decoded_test_data = decoded_test_data.cpu().numpy()
        prediction_class = decoded_test_data.argmax(axis=1)

        for i, pred_trace in enumerate(prediction_class):                   
            selection = invert(pred_trace[start_read:end_read])
            current_mask = invert(batch_y[i, :][start_read:end_read])

            if selection.any() and current_mask.any():
                ntp += 1
            elif selection.any() and not current_mask.any():
                nfp += 1
            elif not selection.any() and current_mask.any():
                nfn += 1
            elif not selection.any() and not current_mask.any():
                ntn += 1

# Compute and display metrics
cm = np.array([[ntp, nfp], [nfn, ntn]])
accuracy = (ntp + ntn) / (ntp + ntn + nfn + nfp)
precision = ntp / (ntp + nfp)
recall = ntp / (ntp + nfn)
f1 = 2 * precision * recall / (precision + recall)

print(cm)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

#plt.show()