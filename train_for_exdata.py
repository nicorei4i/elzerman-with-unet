# Import necessary libraries
#%%
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.sgd
import torch.optim.lr_scheduler as lr_scheduler
from unet_model import UNet
from aenc_model import Conv1DAutoencoder
from torch.utils.data import DataLoader
from dataset import SimDataset, Noise, MinMaxScalerTransform, MeasuredNoise
from HDF5Data import HDF5Data
from sklearn.preprocessing import MinMaxScaler
import time
from test_lib import get_snr, get_scores_unet, save_scores, plot_unet, get_snr_experimental
mpl.rcParams.update({'figure.max_open_warning': 0})

def main():
    # Check if GPU is available
    print('GPU available: ', torch.cuda.is_available())

    # Set device to GPU if available, else CPUsq
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):
        num_workers = 4
    else: 
        num_workers = 1
        mpl.use('TkAgg')


    #device='cpu'
    print(device)

    # Set up directory paths
    current_dir = os.getcwd()  
    current_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(current_dir, 'sim_traces')
    real_data_dir = os.path.join(current_dir, 'real_data')
    file_name = 'sim_read_traces_train_20k_pure'  
    val_name = 'sim_read_traces_val_pure'  
    # file_name = 'sim_read_traces_train_1k'  
    # val_name = 'sim_read_traces_val'  
    
    test_name = 'sliced_traces' 
    test_whole_name = 'sliced_traces_whole' 
    #test_trace_name = 'test_trace'
    test_trace_name = "494_2_3T_elzermann_testtrace_at_b'repitition'_771.000_b'Pulse_for_Qdac - Tburst'_554.500"
    noise_name = '545_1.9T_pg13_vs_tc'

    noise_path = os.path.join(real_data_dir, '{}.hdf5'.format(noise_name))
    hdf5Data_noise = HDF5Data(wdir=trace_dir)
    hdf5Data_noise.set_path(noise_path)

    hdf5Data_noise.set_filename()
    hdf5Data_noise.set_traces()
    hdf5Data_noise.set_measure_data_and_axis()
    
    tc = np.array(hdf5Data_noise.measure_axis[1]).T


    hdf5Data_noise.set_traces_dt() # self.data.set_traces_dt()
    noise_traces = np.array(hdf5Data_noise.traces) #self.data.traces
    mask = np.logical_and(8e-6<tc, tc<12e-6)
    noise_traces = noise_traces[mask]
    print(file_name)
    print(val_name)
    print(test_name)
    print(test_whole_name)
    

    # Construct full paths for the HDF5 files
    hdf5_file_path = os.path.join(trace_dir, '{}.hdf5'.format(file_name))  
    hdf5_file_path_val = os.path.join(trace_dir, '{}.hdf5'.format(val_name))  
    hdf5_file_path_test = os.path.join(real_data_dir, '{}.hdf5'.format(test_name))  
    #hdf5_file_path_whole = os.path.join(trace_dir, '{}.hdf5'.format(test_whole_name))  
    test_trace_path = os.path.join(real_data_dir, '{}.npy'.format(test_trace_name))  

    test_trace = np.load(test_trace_path)
    test_trace = test_trace[:15000]
    test_trace = test_trace[-5000:]

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

    # with h5py.File(hdf5_file_path_whole, 'r') as file:  
    #         all_keys = file.keys()  
    #         whole_data = np.array([file[key] for key in all_keys], dtype=np.float32)  
    #         print(whole_data.shape)  
    # Define parameters for noise and simulation
  
    T = 0.006  
    n_samples = 8192

    train_scaler = MinMaxScalerTransform()
    test_scaler = MinMaxScalerTransform()
    noise_transform = MeasuredNoise(noise_traces=noise_traces)
    def get_loaders(amps, amps_dist):
    # Define parameters for interference signals
        # Create instances of Noise and MinMaxScalerTransform classes
        #noise_transform.set_noise_traces(noise_traces)
        noise_transform.set_amps(amps) 
        noise_transform.set_amps_dist(amps_dist)
        # Fit scalers using data from the HDF5 files
        dataset = SimDataset(hdf5_file_path, scale_transform=None, noise_transform=noise_transform)  
        train_loader = DataLoader(dataset, batch_size=data.shape[0], shuffle=True, num_workers=1, persistent_workers=True, pin_memory=True)
        noisy_data = np.array([batch_x.cpu().numpy() for batch_x, batch_y in train_loader])    
        noisy_data = noisy_data[0, :, 0, :]
        train_scaler.fit_data(noisy_data)


        test_scaler.fit_from_hdf5(hdf5_file_path_test)
        

        batch_size = 32
        # Create instances of SimDataset class for training and validation datasets
        print('Creating datasets...')
        dataset = SimDataset(hdf5_file_path, scale_transform=train_scaler, noise_transform=noise_transform)  
        val_dataset = SimDataset(hdf5_file_path_val, scale_transform=train_scaler, noise_transform=noise_transform)
        test_dataset = SimDataset(hdf5_file_path_test, scale_transform=test_scaler, noise_transform=None)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True)
        
        return train_loader, val_loader, test_loader

    # Create DataLoaders
   

    # Initialize model, loss function, and optimizer
    
    
   
    # Training loop with validation
    def train_model(train_loader, val_loader):
        start = time.time()
        model = UNet().to(device)  
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001) 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
        print('Start training...')
        train_losses = []
        val_losses = []

        num_epochs = 25 
        
        for epoch in range(num_epochs):  
            start_train = time.time()
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
            lr = optimizer.param_groups[0]["lr"]
            # scheduler.step()

            model.eval()
            val_loss = 0.0
            start_val = time.time()
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

            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, lr = {lr}, duration = {(time.time()-start_train):.4f}')
        print()
        print(f"Finished Training in {(time.time() - start):.1f}")
        print()
        return model
    
#     sigma = np.linspace(0, 0.8, 100)    
#     x = np.linspace(-1, 1, 100)
#     weights_sigma = np.exp(0.5*(-((x-0.7)/0.4)**2))
#     weights_sigma /= np.sum(weights_sigma)

#     freqs = [50, 200, 600, 1000]
#     amps = np.array([sigma, sigma, sigma, sigma])
#     print(amps.shape)
#     weights_amps = np.array([weights_sigma, weights_sigma, weights_sigma, weights_sigma])
    amps = np.linspace(2, 3, 100)
    print(f'Noise amps are from {np.min(amps)} to {np.max(amps)}')
    x = np.linspace(-1, 1, len(amps))
    amps_dist = np.exp(0.5*(-((x)/0.5)**2))
    amps_dist /= np.sum(amps_dist)
    # fig, ax = plt.subplots()
    # ax.plot(amps, amps_dist)
    # plt.show()

    train_loader, val_loader, test_loader = get_loaders(amps, amps_dist)
    model = train_model(train_loader, val_loader)
    model.eval()
    
    with torch.no_grad():
        model_dir = os.path.join(current_dir, 'unet_params_ex')
        x, y = next(iter(val_loader))  # Get a batch of validation data
        x = x.to(device)
        y = y.to(device)
        
        decoded_test_data = model(x)
        m = torch.nn.Softmax(dim=1)
        decoded_test_data = m(decoded_test_data)
        decoded_test_data = decoded_test_data.cpu().numpy()  # Get model output for visualization
        prediction_class = decoded_test_data.argmax(axis=1)
        
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        temp_dataset = SimDataset(hdf5_file_path, scale_transform=None, noise_transform=noise_transform)  
        temp_loader = DataLoader(temp_dataset, batch_size=data.shape[0], shuffle=True, num_workers=1, persistent_workers=True, pin_memory=True)
        snr = get_snr(temp_loader)
        for i in range(32):
            fig, axs = plt.subplots(4, 1, figsize=(15, 5), sharex=True)  # Create a figure with 4 subplots
            fig.suptitle(f'Validation Traces (snr = {snr:.2f}dB)')
            axs[1].plot(x[i].reshape(-1), label='Noisy', color='mediumblue', linewidth=0.9)
            axs[1].tick_params(labelbottom=False)
            axs[2].plot(prediction_class[i], label='Denoised', color='mediumblue', linewidth=0.9)
            axs[2].set_ylim(-0.1, 1.1)
            axs[2].tick_params(labelbottom=False)
            axs[3].plot(decoded_test_data[i, 1, :], label='$p(1)$', color='mediumblue', linewidth=0.9)
            axs[3].set_ylim(-0.1, 1.1)
            axs[0].plot(y[i].reshape(-1), label='Clean', color='mediumblue', linewidth=0.9)
            axs[0].set_ylim(-0.1, 1.1)
            axs[0].tick_params(labelbottom=False)
            for ax in axs:
                ax.legend()
                #ax.set_ylim(-0.1, 1.1)
            #plt.show(block=False)
            plt.savefig(os.path.join(model_dir, f'unet_val_{i}.pdf'))  # Save each figure
        print('Figures saved')

        x, y = next(iter(test_loader))  # Get a batch of validation data
        x = x.to(device)
        decoded_test_data = model(x)
        m = torch.nn.Softmax(dim=1)
        decoded_test_data = m(decoded_test_data)
        decoded_test_data = decoded_test_data.cpu().numpy()  # Get model output for visualization
        prediction_class = decoded_test_data.argmax(axis=1)
        x = x.cpu().numpy()
        
        for i in range(32):
            fig, axs = plt.subplots(3, 1, figsize=(15, 5), sharex=True)  # Create a figure with 4 subplots
            fig.suptitle('Validation Traces')
            axs[0].plot(x[i].reshape(-1), label='Noisy', color='mediumblue', linewidth=0.9)
            axs[0].tick_params(labelbottom=False)
            axs[1].plot(prediction_class[i], label='Denoised', color='mediumblue', linewidth=0.9)
            axs[1].tick_params(labelbottom=False)
            axs[1].set_ylim(-0.1, 1.1)
            axs[2].plot(decoded_test_data[i, 1, :], label='$p(1)$', color='mediumblue', linewidth=0.9)
            axs[2].set_ylim(-0.1, 1.1)
            
            for ax in axs:
                ax.legend()
                #ax.set_ylim(-0.1, 1.1)
            #plt.show(block=False)
            plt.savefig(os.path.join(model_dir, f'unet_test_{i}.pdf'))  # Save each figure

        x = torch.tensor(test_scaler(test_trace).reshape(1,1,-1), dtype=torch.float32).to(device)
        decoded_test_data = model(x)
        m = torch.nn.Softmax(dim=1)
        decoded_test_data = m(decoded_test_data)
        decoded_test_data = decoded_test_data.cpu().numpy()  # Get model output for visualization
        prediction_class = decoded_test_data.argmax(axis=1)
        x = x.cpu().numpy()
        

        fig, axs = plt.subplots(3, 1, figsize=(15, 5), sharex=True)  # Create a figure with 4 subplots
        fig.suptitle('Validation Traces')
        axs[0].plot(x.reshape(-1), label='Noisy', color='mediumblue', linewidth=0.9)
        axs[0].tick_params(labelbottom=False)
        axs[1].plot(prediction_class[0], label='Denoised', color='mediumblue', linewidth=0.9)
        axs[1].tick_params(labelbottom=False)
        axs[2].plot(decoded_test_data[0, 1, :], label='$p(1)$', color='mediumblue', linewidth=0.9)
        for ax in axs:
            ax.legend()
            #ax.set_ylim(-0.1, 1.1)
        #plt.show(block=False)
        plt.savefig(os.path.join(model_dir, f'unet_test_trace.pdf'))  # Save each figure

            
        print('Figures saved')



        

    print('Saving model parameters...')
    os.makedirs(model_dir, exist_ok=True)  
    state_dict_name = 'model_weights'  
    state_dict_path = os.path.join(model_dir, '{}.pth'.format(state_dict_name))  
    torch.save(model.state_dict(), state_dict_path)  
    print('done')
if __name__ == '__main__':
    main()