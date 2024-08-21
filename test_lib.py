#%%
import numpy as np
import os
import h5py
import matplotlib
matplotlib.use('Agg')
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # Directory setup
# current_dir = os.path.dirname(os.path.abspath(__file__))
# trace_dir = os.path.join(current_dir, 'traces')
# file_name = 'sim_elzerman_traces_test'  # Name for the test HDF5 file
# #mask_name = 'sim_elzerman_test_masks'  # Name for the mask HDF5 file

# # Construct full paths for the HDF5 files
# hdf5_file_path = os.path.join(trace_dir, '{}.hdf5'.format(file_name))  # Test file path
# #hdf5_file_path_masks = os.path.join(current_dir, '{}.hdf5'.format(mask_name))  # Mask file path

# # Read data from the test HDF5 file
# with h5py.File(hdf5_file_path, 'r') as file:
#     all_keys = file.keys()  # Get all keys (datasets) in the HDF5 file
#     test_data = np.array([file[key] for key in all_keys], dtype=np.float32)  # Read data and store in a numpy array
#     print(test_data.shape)  # Print the data shape for verification

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

# def get_loaders(s):
# # Define parameters for interference signals
#     print('Noise Amp: ', s)
#     interference_amps = np.ones(4) * s  
#     interference_freqs = [50, 200, 600, 1000]  

#     # Create instances of Noise and MinMaxScalerTransform classes
#     noise_transform = Noise(n_samples, T, s, interference_amps, interference_freqs)
#     scaler = MinMaxScalerTransform()
    
#     # Fit scalers using data from the HDF5 files
#     scaler.fit_from_hdf5(hdf5_file_path)

#     # Create instances of SimDataset class for training and validation datasets
#     print('Creating datasets...')
#     dataset = SimDataset(hdf5_file_path, scale_transform=scaler, noise_transform=noise_transform)  
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     return loader

# # Model setup
# current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
# model_dir = os.path.join(current_dir, 'unet_weights')  # Directory for model weights
# state_dict_name = 'model_weights'  # Name for the model state dictionary
# state_dict_path = os.path.join(model_dir, '{}.pth'.format(state_dict_name))  # Full path for saving model weights

# # Load the model
# model = UNet().to(device)
# model.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
# model.eval()
def format_number(num):
    try:
        # Attempt to convert num to a float
        num = float(num)
    except ValueError:
        raise TypeError("The input must be a number or a string representing a number.")
    
    # Round the number to 2 decimal places
    rounded_num = round(num, 2)
    
    # Convert the number to a string and replace the decimal dot with an underscore
    formatted_str = str(rounded_num).replace('.', '_')
    
    return formatted_str

def plot_unet(model, test_loader, model_dir, snr):
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
    
    snr = format_number(snr)
    for i in range(2):
        fig, axs = plt.subplots(4, 1, figsize=(15, 5), sharex=True)  # Create a figure with 4 subplots

        fig.suptitle('Validation Traces')

        axs[1].plot(x[i].reshape(-1), label='Noisy', color='mediumblue', linewidth=0.9)
        axs[1].tick_params(labelbottom=False)
        axs[1].legend()

        axs[2].plot(prediction_class[i], label='Denoised', color='mediumblue', linewidth=0.9)
        axs[2].tick_params(labelbottom=False)
        axs[2].legend()

        axs[3].plot(decoded_test_data[i, 1, :], label='$p(1)$', color='mediumblue', linewidth=0.9)
        axs[3].legend()

        axs[0].plot(y[i].reshape(-1), label='Clean', color='mediumblue', linewidth=0.9)
        axs[0].tick_params(labelbottom=False)
        axs[0].legend()
        #plt.show(block=False)
        
        plt.savefig(os.path.join(model_dir, f'unet_{snr}_{i}.pdf'))  # Save each figure


def plot_aenc(model, test_loader, model_dir, snr):
    # Visualize validation dataset predictions
    x, y = next(iter(test_loader))  # Get a batch of validation data
    x = x.to(device)
    y = y.to(device)
    model.eval()
    with torch.no_grad():  # Disable gradient calculation for visualization
        decoded_test_data = model(x)
        # m = torch.nn.Softmax(dim=1)
        # decoded_test_data = m(decoded_test_data)
        # decoded_test_data = decoded_test_data.cpu().numpy()  # Get model output for visualization
        # prediction_class = decoded_test_data.argmax(axis=1)
        
        x = x.cpu()
        y = y.cpu()
        decoded_test_data=decoded_test_data.cpu().numpy()

    # Plot the results
    print('Plotting the results...')
    snr = format_number(snr)
    for i in range(2):
        fig, axs = plt.subplots(3, 1, figsize=(15, 5), sharex=True)  # Create a figure with 4 subplots

        fig.suptitle('Validation Traces')

        axs[1].plot(x[i].numpy().reshape(-1), label='Noisy', color='mediumblue', linewidth=0.9)
        axs[1].tick_params(labelbottom=False)
        axs[1].legend()

        #axs[2].plot(decoded_test_data[i, 1, :], label='$p(1)$', color='mediumblue', linewidth=0.9)
        axs[2].plot(decoded_test_data[i, 0, :], label='denoised', color='mediumblue', linewidth=0.9)
        axs[2].legend()

        axs[0].plot(y[i].numpy().reshape(-1), label='Clean', color='mediumblue', linewidth=0.9)
        axs[0].tick_params(labelbottom=False)
        axs[0].legend()
        #plt.show(block=False)
       
        plt.savefig(os.path.join(model_dir, f'aenc_{snr}_{i}.pdf'))  # Save each figure


def get_snr(loader):
    x, y = next(iter(loader))  # Get a batch of validation data
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    x = x.squeeze(1)
    y = y.squeeze(1)

    signals = y
    noise = x-y

    signal_powers = np.mean(signals**2, axis=1)
    noise_powers = np.mean(noise**2, axis=1)


    snr = np.mean(1/noise_powers)

    snr = 10* np.log10(snr)

    return snr


def get_snr_experimental(clean_trace, noisy_trace, scaler):
    clean_trace = scaler(clean_trace)
    clean_trace = clean_trace.reshape(-1)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(clean_trace)
    # ax.plot(noisy_trace)
    # plt.show(block=True)

    signals = clean_trace
    noise = noisy_trace-clean_trace

    signal = np.max(clean_trace) - np.min(clean_trace)
    signal = signal**2
    noise_powers = np.mean(noise**2)


    snr = np.mean(signal/noise_powers)

    snr = 10* np.log10(snr)

    return snr


def invert(arr):
    where_0 = np.where(arr == 0)
    where_1 = np.where(arr == 1)
    arr[where_0] = 1
    arr[where_1] = 0
    return np.array(arr, dtype=np.int32)


def get_scores_unet(model, test_loader):
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
            #prob_1 = decoded_test_data[:, 1, :]
            prediction_class = decoded_test_data.argmax(axis=1)
            #prediction_class = decoded_test_data.numpy().squeeze(1)
            for i, pred_trace in enumerate(prediction_class):                   
                #selection = [prob[start_read:end_read]< 0.1]
                selection = [pred_trace[start_read:end_read] == 0]
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
        #accuracy = (ntp + ntn) / (ntp + ntn + nfn + nfp)
        if ntp + nfp == 0: 
            precision = np.nan
        else:
            precision = ntp / (ntp + nfp)
        if ntp + nfp == 0:
            recall = np.nan
        else:
            recall = ntp / (ntp + nfn)
        #f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, cm 
    
def get_scores_aenc(model, test_loader):
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
            # m = torch.nn.Softmax(dim=1)
            # decoded_test_data = m(decoded_test_data)
            # decoded_test_data = decoded_test_data.cpu().numpy()
            # prediction_class = decoded_test_data.argmax(axis=1)
            prediction_class = decoded_test_data.numpy().squeeze(1)
            for i, pred_trace in enumerate(prediction_class):                   
                #selection = invert(pred_trace[start_read:end_read])
                selection = [pred_trace[start_read:end_read] < 0.01]
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
        #accuracy = (ntp + ntn) / (ntp + ntn + nfn + nfp)
        if ntp + nfp == 0: 
            precision = np.nan
        else:
            precision = ntp / (ntp + nfp)
        if ntp + nfp == 0:
            recall = np.nan
        else:
            recall = ntp / (ntp + nfn)
        #f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, cm     
# cms = []
# precisions = []
# recalls = []
# snrs = []
# interference_freqs = [50, 200, 600, 1000]  
# noise_sigs = np.linspace(0.01, 2, 10)
# for s in noise_sigs: 
#     loader = get_loaders(s)
#     #plot(loader)
#     interference_amps = np.ones(4) * s  
#     snrs.append(get_snr(T, s, interference_amps, interference_freqs))
#     score = get_scores(loader)
#     print(score)
#     precisions.append(score[0])
#     recalls.append(score[1])
#     cms.append(score[2])

# precisions = np.array(precisions)
# recalls = np.array(recalls)
# cms = np.array(cms)

# #%%
# fig, ax = plt.subplots(1, 1)
# ax.scatter(snrs, precisions, label='precision')
# ax.scatter(snrs, recalls, label='recall')
# #ax.set_xscale('log')
# #ax.set_yscale('log')
def save_scores(snrs, precisions, recalls, cms, pickle_name):
    scores = {
        'snr':snrs,
        'precision': precisions,
        'recall': recalls,
        'cm': cms
    }
    current_dir = os.path.dirname(os.path.abspath(__file__))

    #pickle_name = 'unet_scores'
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

# ax.legend()
# plt.show()
# print(snrs)
# print(scores)
