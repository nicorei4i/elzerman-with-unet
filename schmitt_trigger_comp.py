#%%
# Import necessary libraries
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.sgd
import torch.optim.lr_scheduler as lr_scheduler
from aenc_model import Conv1DAutoencoder
from torch.utils.data import DataLoader
from dataset import SimDataset, Noise, MinMaxScalerTransform
from sklearn.preprocessing import MinMaxScaler
import time
from test_lib import get_snr, get_scores_aenc, save_scores, plot_aenc, plot_schmitt, get_scores_schmitt

def main():
    # Set up directory paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(current_dir, 'sim_traces')
    test_name = 'sim_elzerman_traces_test_100'  

    print(test_name)

    # Construct full paths for the HDF5 files
    hdf5_file_path_test = os.path.join(trace_dir, '{}.hdf5'.format(test_name))  

    # Read data from the training HDF5 file

    # Read data from the validation HDF5 file
    with h5py.File(hdf5_file_path_test, 'r') as file:  
            all_keys = file.keys()  
            test_data = np.array([file[key] for key in all_keys], dtype=np.float32)  
            print(test_data.shape)  

    # Define parameters for noise and simulation
  
    T = 0.006  
    n_samples = 8192


    def get_loaders(s):
    # Define parameters for interference signals
        print('Noise Sigma: ', s)
        interference_amps = np.ones(4) * s  
        interference_freqs = [50, 200, 600, 1000]  

        # Create instances of Noise and MinMaxScalerTransform classes
        noise_transform = Noise(n_samples, T, s, interference_amps, interference_freqs)

        batch_size = 32  
        # Create instances of SimDataset class for training and validation datasets
        print('Creating datasets...')
        test_dataset = SimDataset(hdf5_file_path_test, scale_transform=False, noise_transform=noise_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        
        return test_loader

    cms = []
    precisions = []
    recalls = []
    snrs = []
    noise_sigs = np.linspace(0.05, 0.2, 10)
    print('noise sigs: ', noise_sigs)
    for s in noise_sigs: 
        test_loader = get_loaders(s)
        with torch.no_grad():
            snr = get_snr(test_loader)
            model_dir = os.path.join(current_dir, 'schmitt_results')
            plot_schmitt(test_loader, model_dir, snr)
            score = get_scores_schmitt(test_loader)
            print('snr: ', snr)
            print('score: ', score)
            print('')
            snrs.append(snr)
            precisions.append(score[0])
            recalls.append(score[1])
            cms.append(score[2])

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    cms = np.array(cms)

    save_scores(snrs, precisions, recalls, cms, 'schmitt_scores')

    
if __name__ == '__main__':
    main()