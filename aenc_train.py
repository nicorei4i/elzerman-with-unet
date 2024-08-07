# Import necessary libraries
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.sgd
from aenc_model import Conv1DAutoencoder
from torch.utils.data import DataLoader
from dataset import SimDataset, Noise, MinMaxScalerTransform
from sklearn.preprocessing import MinMaxScaler
import time
from test_lib import get_snr, get_scores_aenc, save_scores, plot_aenc

def main():
    # Check if GPU is available
    print('GPU available: ', torch.cuda.is_available())

    # Set device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Set up directory paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(current_dir, 'traces')
    file_name = 'sim_elzerman_traces_train'  
    val_name = 'sim_elzerman_traces_val'  
    test_name = 'sim_elzerman_traces_test'  

    # Construct full paths for the HDF5 files
    hdf5_file_path = os.path.join(trace_dir, '{}.hdf5'.format(file_name))  
    hdf5_file_path_val = os.path.join(trace_dir, '{}.hdf5'.format(val_name))  
    hdf5_file_path_test = os.path.join(trace_dir, '{}.hdf5'.format(test_name))  

    # Read data from the training HDF5 file
    with h5py.File(hdf5_file_path, 'r') as file:  
        all_keys = file.keys()  
        data = np.array([file[key] for key in all_keys],dtype=np.float32)  
        print(data.shape)  

    # Read data from the validation HDF5 file
    with h5py.File(hdf5_file_path_val, 'r') as file:  
        all_keys = file.keys()  
        val_data = np.array([file[key] for key in all_keys], dtype=np.float32)  
        print(val_data.shape)  

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
        train_scaler = MinMaxScalerTransform()
        val_scaler = MinMaxScalerTransform()
        test_scaler = MinMaxScalerTransform()

        # Fit scalers using data from the HDF5 files
        train_scaler.fit_from_hdf5(hdf5_file_path)
        val_scaler.fit_from_hdf5(hdf5_file_path_val)
        test_scaler.fit_from_hdf5(hdf5_file_path_test)
        

        batch_size = 32  
        # Create instances of SimDataset class for training and validation datasets
        print('Creating datasets...')
        dataset = SimDataset(hdf5_file_path, scale_transform=train_scaler, noise_transform=noise_transform)  
        val_dataset = SimDataset(hdf5_file_path_val, scale_transform=val_scaler, noise_transform=noise_transform)
        test_dataset = SimDataset(hdf5_file_path_test, scale_transform=test_scaler, noise_transform=noise_transform)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        
        return train_loader, val_loader, test_loader

    # Create DataLoaders

    # Initialize model, loss function, and optimizer
    model = Conv1DAutoencoder().to(device)  
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)  

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
                batch_y = batch_y.to(device)
                #.squeeze(1).long()

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
                    batch_y = batch_y.to(device)
                    #.squeeze(1).long()
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
        
    
    cms = []
    precisions = []
    recalls = []
    snrs = []
    noise_sigs = np.linspace(0.01, 0.7, 10)
    print('noise sigs: ', noise_sigs)
    for s in noise_sigs: 
        train_loader, val_loader, test_loader = get_loaders(s)
        train_model(train_loader, val_loader)
        model.eval()
        with torch.no_grad():
            snr = get_snr(test_loader)
            model_dir = os.path.join(current_dir, 'aenc_weights')
            plot_aenc(model, test_loader, model_dir, snr)
            score = get_scores_aenc(model, test_loader)
            print('snr: ', snr)
            print('score: ', score)
            snrs.append(snr)
            precisions.append(score[0])
            recalls.append(score[1])
            cms.append(score[2])

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    cms = np.array(cms)

    save_scores(snrs, precisions, recalls, cms, 'aenc_scores')

    print('Saving model parameters...')
    
    os.makedirs(model_dir, exist_ok=True)  
    state_dict_name = 'model_weights'  
    state_dict_path = os.path.join(model_dir, '{}.pth'.format(state_dict_name))  
    torch.save(model.state_dict(), state_dict_path)  
    print('done')
if __name__ == '__main__':
    main()