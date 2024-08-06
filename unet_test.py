#%%
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from unet_model import UNet
from aenc_model import Conv1DAutoencoder
from torch.utils.data import DataLoader
from dataset import SimDataset, Noise, MinMaxScalerTransform
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pickle
from pathlib import Path


print('GPU available: ', torch.cuda.is_available())

# Set device to GPU if available, otherwise use CPU
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'
# Directory setup
current_dir = os.path.dirname(os.path.abspath(__file__))
trace_dir = os.path.join(current_dir, 'traces')
file_name = 'sim_elzerman_traces_test'  # Name for the test HDF5 file
#mask_name = 'sim_elzerman_test_masks'  # Name for the mask HDF5 file

# Construct full paths for the HDF5 files
hdf5_file_path = os.path.join(trace_dir, '{}.hdf5'.format(file_name))  # Test file path
#hdf5_file_path_masks = os.path.join(current_dir, '{}.hdf5'.format(mask_name))  # Mask file path

# Read data from the test HDF5 file
with h5py.File(hdf5_file_path, 'r') as file:
    all_keys = file.keys()  # Get all keys (datasets) in the HDF5 file
    test_data = np.array([file[key] for key in all_keys], dtype=np.float32)  # Read data and store in a numpy array
    print(test_data.shape)  # Print the data shape for verification

# Define time indices
t_L, t_W, t_R, t_U = 0.5e-3, 0.0, 1.0e-3, 1.5e-3  # Time intervals in seconds
times = np.array([t_L, t_W, t_R, t_U])
trace_duration = np.sum(times)
times = np.array([t_L, t_L + t_W, t_L + t_W + t_R, t_L + t_W + t_R + t_U])
n_samples = 8192
times_indices = times * n_samples / trace_duration
times_indices = times_indices.astype(np.int64)
start_read, end_read = times_indices[1], times_indices[2]

# Define parameters for noise and simulation

T = 0.006  
N = 2
n_samples = 8192
dt = T / n_samples
n_cycles = 2  
batch_size = 32

def get_loaders(s):
# Define parameters for interference signals
    print('Noise Amp: ', s)
    interference_amps = np.ones(4) * s  
    interference_freqs = [50, 200, 600, 1000]  

    # Create instances of Noise and MinMaxScalerTransform classes
    noise_transform = Noise(n_samples, T, s, interference_amps, interference_freqs)
    scaler = MinMaxScalerTransform()
    
    # Fit scalers using data from the HDF5 files
    scaler.fit_from_hdf5(hdf5_file_path)

    # Create instances of SimDataset class for training and validation datasets
    print('Creating datasets...')
    dataset = SimDataset(hdf5_file_path, scale_transform=scaler, noise_transform=noise_transform)  
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# Model setup
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
model_dir = os.path.join(current_dir, 'unet_weights')  # Directory for model weights
state_dict_name = 'model_weights'  # Name for the model state dictionary
state_dict_path = os.path.join(model_dir, '{}.pth'.format(state_dict_name))  # Full path for saving model weights

# Load the model
model = UNet().to(device)
model.load_state_dict(torch.load(state_dict_path, map_location=device))
model.eval()

def plot(test_loader):
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
        
        x = x.cpu().numpy()
        y = y.cpu().numpy()

    # Plot the results
    print('Plotting the results...')
    print('x: ', type(x))
    print('y: ', type(y))
    print('prediction class: ', type(prediction_class))
    print('decoded test data: ', type(decoded_test_data))
    
    
    for i in range(2):
        fig, axs = plt.subplots(4, 1, figsize=(15, 5), sharex=True)  # Create a figure with 4 subplots

        fig.suptitle('Validation Traces')

        axs[0].plot(x[i].reshape(-1), label='Noisy', color='mediumblue', linewidth=0.9)
        axs[0].tick_params(labelbottom=False)
        axs[0].legend()

        axs[1].plot(prediction_class[i], label='Denoised', color='mediumblue', linewidth=0.9)
        axs[1].tick_params(labelbottom=False)
        axs[1].legend()

        axs[2].plot(decoded_test_data[i, 1, :], label='$p(1)$', color='mediumblue', linewidth=0.9)
        axs[2].tick_params(labelbottom=False)
        axs[2].legend()

        axs[3].plot(y[i].reshape(-1), label='Clean', color='mediumblue', linewidth=0.9)
        axs[3].legend()
        #plt.show(block=False)
        plt.savefig(os.path.join(model_dir, f'validation_trace_{i}.png'))  # Save each figure


def get_snr(sim_t, sigma, amps, freqs):

    shape = 8192
    T = sim_t
    amps = amps
    freqs = freqs
    white_sigma = sigma
    pink_sigma = 0.1 * sigma


    white_noise = np.random.normal(0.0, white_sigma, shape)

    exponents = np.fft.fftfreq(shape)
    exponents[0] = 1  # Avoid division by zero
    amplitudes = 1 / np.sqrt(np.abs(exponents))
    amplitudes[0] = 0  # Set the DC component to 0
    random_phases = np.exp(2j * np.pi * np.random.random(shape))
    pink_noise_spectrum = amplitudes * random_phases
    pink_noise = np.fft.ifft(pink_noise_spectrum).real

    interference_noise = 0
    t = np.linspace(0, T, shape)
    phi_0 = np.random.uniform(0, 2 * np.pi, 1)  # Random initial phase

    for i, amp in enumerate(amps):
        interference_noise += (amp * np.sin(2 * np.pi * freqs[i] * t + phi_0) + white_noise + pink_sigma * pink_noise)
        
        
    noise = white_noise + pink_noise + interference_noise
    snr = 2/np.std(noise)

    return snr

def invert(arr):
    where_0 = np.where(arr == 0)
    where_1 = np.where(arr == 1)
    arr[where_0] = 1
    arr[where_1] = 0
    return arr


def get_scores(test_loader):
    # Initialize confusion matrix components
    nfn, nfp, ntn, ntp = 0, 0, 0, 0
    # Validate the model
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch_x, batch_y in test_loader:  # Loop over each batch of validation data
            batch_x = batch_x.to(device)
            decoded_test_data = model(batch_x)
            decoded_test_data = decoded_test_data.cpu()
            batch_y = batch_y.numpy()
            batch_y = batch_y.squeeze(1)
            m = torch.nn.Softmax(dim=1)
            decoded_test_data = m(decoded_test_data)
            decoded_test_data = decoded_test_data.cpu().numpy()
            prediction_class = decoded_test_data.argmax(axis=1)
            # prediction_class = decoded_test_data.numpy().squeeze(1)
            for i, pred_trace in enumerate(prediction_class):                   
                selection = invert(pred_trace[start_read:end_read])
                # selection = [pred_trace[start_read:end_read] < 0.001]
                selection = np.array(selection)
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
        return precision, recall, cm 
cms = []
precisions = []
recalls = []
snrs = []
interference_freqs = [50, 200, 600, 1000]  
noise_sigs = np.linspace(0.01, 2, 20)
for s in noise_sigs: 
    print(s)
    loader = get_loaders(s)
    plot(loader)
    interference_amps = np.ones(4) * s  
    snrs.append(get_snr(T, s, interference_amps, interference_freqs))
    score = get_scores(loader)
    print(score)
    precisions.append(score[0])
    recalls.append(score[1])
    cms.append(score[2])

precisions = np.array(precisions)
recalls = np.array(recalls)
cms = np.array(cms)

#%%
fig, ax = plt.subplots(1, 1)
ax.scatter(snrs, precisions, label='precision')
ax.scatter(snrs, recalls, label='recall')
#ax.set_xscale('log')
#ax.set_yscale('log')

scores = {
    'snr':snrs,
    'precision': precisions,
    'recall': recalls,
    'cm': cms
}


pickle_name = 'unet_scores'
pickle_dir = os.path.join(current_dir, 'results')  # Directory for model weights
os.makedirs(pickle_dir, exist_ok=True)  
pickle_path = os.path.join(pickle_dir, '{}.pkl'.format(pickle_name))  
pickle_path = Path(pickle_path)

def check_and_rename(file_path: Path, add: int = 0) -> Path:
    original_file_path = file_path
    if add != 0:
        file_path = file_path.with_stem(file_path.stem + "_" + str(add))
    if not os.path.isfile(file_path):
        return file_path
    else:
        return check_and_rename(original_file_path, add + 1)


pickle_path = check_and_rename(pickle_path, add=0)
with open(pickle_path, 'wb') as f:
            pickle.dump(scores, f)

ax.legend()
plt.show()
print(snrs)
print(scores)
