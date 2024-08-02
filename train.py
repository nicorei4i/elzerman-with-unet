# Import necessary libraries
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.sgd
from model import UNet
from torch.utils.data import DataLoader
from dataset import SimDataset, Noise, MinMaxScalerTransform
from sklearn.preprocessing import MinMaxScaler
import time
def main():
    # Check if GPU is available
    print('GPU available: ', torch.cuda.is_available())

    # Set device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Set up directory paths
    current_dir = os.getcwd()  
    file_name = 'sim_elzerman_traces_train_10k'  
    val_name = 'sim_elzerman_traces_val'  

    # Construct full paths for the HDF5 files
    hdf5_file_path = os.path.join(current_dir, '{}.hdf5'.format(file_name))  
    hdf5_file_path_val = os.path.join(current_dir, '{}.hdf5'.format(val_name))  

    # Read data from the training HDF5 file
    with h5py.File(hdf5_file_path, 'r') as file:  
        all_keys = file.keys()  
        data = np.array([file[key] for key in all_keys])  
        print(data.shape)  

    # Read data from the validation HDF5 file
    with h5py.File(hdf5_file_path_val, 'r') as file:  
        all_keys = file.keys()  
        val_data = np.array([file[key] for key in all_keys])  
        print(val_data.shape)  

    # Define parameters for noise and simulation
    noise_std = 0.3  
    T = 0.006  
    n_samples = 8192
    dt = T / n_samples
    n_cycles = 2  


    def get_loaders(s):
    # Define parameters for interference signals
        print('Noise Sigma: ', s)
        interference_amps = np.ones(4) * s  
        interference_freqs = [50, 200, 600, 1000]  

        # Create instances of Noise and MinMaxScalerTransform classes
        noise_transform = Noise(n_samples, T, noise_std, interference_amps, interference_freqs)
        train_scaler = MinMaxScalerTransform()
        val_scaler = MinMaxScalerTransform()

        # Fit scalers using data from the HDF5 files
        train_scaler.fit_from_hdf5(hdf5_file_path)
        val_scaler.fit_from_hdf5(hdf5_file_path_val)

        # Create instances of SimDataset class for training and validation datasets
        print('Creating datasets...')
        dataset = SimDataset(hdf5_file_path, scale_transform=train_scaler, noise_transform=noise_transform)  
        val_dataset = SimDataset(hdf5_file_path_val, scale_transform=val_scaler, noise_transform=noise_transform)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        return train_loader, val_loader

    # Create DataLoaders
    batch_size = 32  

    # Initialize model, loss function, and optimizer
    model = UNet().to(device)  
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)  


    def train_model(train_loader, val_loader):
        # Training loop with validation
        print('Start training...')
        train_losses = []
        val_losses = []

        num_epochs = 25
        start = time.time()
        for epoch in range(num_epochs):  
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:  
                optimizer.zero_grad()
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).squeeze(1).long()

                output = model(batch_x)  
                loss = criterion(output, batch_y)  
                loss.backward()  
                optimizer.step()  

                train_loss += loss.item() * batch_x.size(0)
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():  
                for batch_x, batch_y in val_loader:  
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device).squeeze(1).long()
                    output = model(batch_x)  
                    loss = criterion(output, batch_y)  
                    val_loss += loss.item() * batch_x.size(0)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            # train_line.set_ydata(train_losses)
            # train_line.set_xdata(np.arange(1, epoch+2, 1))
            # val_line.set_ydata(val_losses)
            # val_line.set_xdata(np.arange(1, epoch+2, 1))
            # ax.relim()
            # ax.autoscale_view()
            # fig.canvas.draw()
            # fig.canvas.flush_events() 

            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        print()
        print(f"Finished Training in {(time.time() - start):.1f}")
        print()

    noise_sigs = np.linspace(0.1, 0.5, 5)
    print('noise sigs: ', noise_sigs)
    for s in noise_sigs: 
        train_loader, val_loader = get_loaders(s)
        train_model(train_loader, val_loader)

    print('Saving model parameters...')
    model_dir = os.path.join(current_dir, 'batchsnr10k')
    os.makedirs(model_dir, exist_ok=True)  
    state_dict_name = 'model_weights'  
    state_dict_path = os.path.join(model_dir, '{}.pth'.format(state_dict_name))  
    torch.save(model.state_dict(), state_dict_path)  
    print('done')
if __name__ == '__main__':
    main()