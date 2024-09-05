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
from dataset import SimDataset, Noise, MinMaxScalerTransform, get_real_noise_traces, MeasuredNoise
from sklearn.preprocessing import MinMaxScaler
import time
from test_lib import get_snr, get_scores_aenc, save_scores, plot_aenc

def main():
    real_noise_switch = True
    if real_noise_switch:
        print('Using real noise')
    else:
        print('Using simulated noise') 

    # Check if GPU is available
    print('GPU available: ', torch.cuda.is_available())

    # Set device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Set up directory paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(current_dir, 'sim_traces')
    file_name = 'sim_elzerman_traces_train_1k'  
    val_name = 'sim_elzerman_traces_val'  
    test_name = 'sim_elzerman_traces_test_1k'  

    print(file_name)
    print(val_name)
    print(test_name)

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


    def get_loaders_simnoise(s):
    # Define parameters for interference signals
        print('Noise Sigma: ', s)
        interference_amps = np.ones(4) * s  
        interference_freqs = [50, 200, 600, 1000]  

        # Create instances of Noise and MinMaxScalerTransform classes
        noise_transform = Noise(n_samples, T, s, interference_amps, interference_freqs)

        batch_size = 32  
        # Create instances of SimDataset class for training and validation datasets
        print('Creating datasets...')
        dataset = SimDataset(hdf5_file_path, scale_transform=None, noise_transform=noise_transform)  
        val_dataset = SimDataset(hdf5_file_path_val, scale_transform=None, noise_transform=noise_transform)
        test_dataset = SimDataset(hdf5_file_path_test, scale_transform=None, noise_transform=noise_transform)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        
        return train_loader, val_loader, test_loader


    def get_loaders_realnoise(s):
    # Define parameters for interference signals
        print('Noise Amp: ', s)
        
        noise_traces = get_real_noise_traces()
        noise_traces_train = noise_traces[:-5]
        noise_traces_val = noise_traces[-5:]
        # Create instances of Noise and MinMaxScalerTransform classes
        noise_transform_train = MeasuredNoise(noise_traces=noise_traces_train, amps=[s],amps_dist=[1])
        noise_transform_val = MeasuredNoise(noise_traces=noise_traces_val, amps=[s],amps_dist=[1])
        
        batch_size = 32  
        # Create instances of SimDataset class for training and validation datasets
        print('Creating datasets...')
        dataset = SimDataset(hdf5_file_path, scale_transform=None, noise_transform=noise_transform_train)  
        val_dataset = SimDataset(hdf5_file_path_val, scale_transform=None, noise_transform=noise_transform_val)
        test_dataset = SimDataset(hdf5_file_path_test, scale_transform=None, noise_transform=noise_transform_val)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        
        return train_loader, val_loader, test_loader


    # Create DataLoaders

    # Initialize model, loss function, and optimizer
   

    def train_model(train_loader, val_loader):

        # Training loop with validation
        print('Start training...')

        model = Conv1DAutoencoder().to(device)  
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

        #criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)  
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=25)

        train_losses = []
        val_losses = []

        num_epochs = 25
        start = time.time()
        for epoch in range(num_epochs): 
            start_train = time.time() 
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
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

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

            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, lr = {lr:.4f}, duration = {(time.time()-start_train):.4f}')
        print()
        print(f"Finished Training in {(time.time() - start):.1f}")
        print()
        return model
        
    
    cms = []
    precisions = []
    recalls = []
    snrs = []
    if real_noise_switch:
        noise_sigs = np.linspace(0.5, 5, 2)
    else: 
        noise_sigs = np.linspace(0.1, 0.8, 2)


    print('noise sigs: ', noise_sigs)
    for s in noise_sigs: 
        if real_noise_switch:
            train_loader, val_loader, test_loader = get_loaders_realnoise(s)
        else:
            train_loader, val_loader, test_loader = get_loaders_simnoise(s)
        
        model = train_model(train_loader, val_loader)
        model.eval()
        with torch.no_grad():
            snr = get_snr(test_loader)
            if real_noise_switch:   
                model_dir = os.path.join(current_dir, 'aenc_weights_real_noise')
            else: 
                model_dir = os.path.join(current_dir, 'aenc_weights_sim_noise')
            
            plot_aenc(model, test_loader, model_dir, snr)
            score_time_start = time.time()
            score = get_scores_aenc(model, test_loader)
            print(f'Time for scoring: {time.time()- score_time_start}s')
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

    if real_noise_switch:
        save_scores(snrs, precisions, recalls, cms, 'aenc_scores_real_noise')
    else: 
        save_scores(snrs, precisions, recalls, cms, 'aenc_scores_sim_noise')

    print('Saving model parameters...')
    
    os.makedirs(model_dir, exist_ok=True)  
    state_dict_name = 'model_weights'  
    state_dict_path = os.path.join(model_dir, '{}.pth'.format(state_dict_name))  
    torch.save(model.state_dict(), state_dict_path)  
    print('done')

if __name__ == '__main__':
    global_start = time.time()
    main()
    print(f'Total duration: {time.time()- global_start}s')
