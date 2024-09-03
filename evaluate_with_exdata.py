
#%%
import numpy as np
import matplotlib as mpl
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle 
import os
from torch.utils.data import DataLoader
from test_lib import schmitt_trigger, get_scores_unet
from unet_model import UNet
import torch
import h5py
from dataset import subtract_lb
from scipy.signal import welch
from HDF5Data import HDF5Data
from dataset import MeasuredNoise, MinMaxScalerTransform, subtract_lb
from sklearn.preprocessing import RobustScaler
import scipy as sc


print('GPU available: ', torch.cuda.is_available())

# Set device to GPU if available, else CPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == torch.device('cuda'):
    num_workers = 4
else: 
    num_workers = 1
    mpl.use('Qt5Agg')

current_dir = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(current_dir, 'unet_params_ex')
ex_data_dir = os.path.join(current_dir, 'real_data')
trace_dir = os.path.join(current_dir, 'sim_traces')

# file_name = 'sim_read_traces_train_10k'  
# test_name = 'sliced_traces' 

# hdf5_file_path_sim = os.path.join(trace_dir, '{}.hdf5'.format(file_name))  
# hdf5_file_path_test = os.path.join(ex_data_dir, '{}.hdf5'.format(test_name))  



# noise_name = '545_1.9T_pg13_vs_tc'

# noise_path = os.path.join(ex_data_dir, '{}.hdf5'.format(noise_name))
# hdf5Data_noise = HDF5Data(wdir=trace_dir)
# hdf5Data_noise.set_path(noise_path)

# hdf5Data_noise.set_filename()
# hdf5Data_noise.set_traces()
# hdf5Data_noise.set_measure_data_and_axis()

# tc = np.array(hdf5Data_noise.measure_axis[1]).T


# hdf5Data_noise.set_traces_dt() # self.data.set_traces_dt()
# noise_traces = np.array(hdf5Data_noise.traces) #self.data.traces
# mask = np.logical_and(8e-6<tc, tc<12e-6)
# noise_traces = noise_traces[mask]
# print(f'Noise traces shape:{noise_traces.shape}')

# noise_transform = MeasuredNoise(noise_traces=noise_traces, amps=[4], amps_dist=[1])


# with h5py.File(hdf5_file_path_sim, 'r') as file:  
#     all_keys = file.keys()  
#     sim_data = np.array([noise_transform(file[key]) for key in all_keys],dtype=np.float32)  
#     print(f'Training traces shape:{sim_data.shape}')


# with h5py.File(hdf5_file_path_test, 'r') as file:  
#     all_keys = file.keys()  
#     test_data = np.array([subtract_lb(file[key]) for key in all_keys],dtype=np.float32)  
#     print(f'Test traces shape:{test_data.shape}')


# sim_scaler = RobustScaler()
# test_scaler = RobustScaler()


# sim_scaler.fit(sim_data)
# test_scaler.fit(test_data)

# sim_data = sim_scaler.transform(sim_data)
# test_data = test_scaler.transform(test_data)

# sim_psd = np.empty_like(welch(sim_data[0])[1])
# for trace in sim_data:
#     f_sim, Pxx = welch(trace)
#     sim_psd += Pxx


# test_psd = np.empty_like(welch(sim_data[0])[1])
# for trace in test_data:
#     f_test, Pxx = welch(trace)
#     test_psd += Pxx

# sim_psd  = sim_psd/ sim_data.shape[0]
# test_psd/= test_data.shape[0]

# plt.plot(f_sim, sim_psd, color='red', alpha=0.5)
# plt.plot(f_sim, test_psd, color='blue', alpha=0.5)
# plt.show()


#%%
def unpickle_loader(name, shuffle=True):
    pickle_path = os.path.join(model_dir, f'{name}.pkl')
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        new_dataloader = DataLoader(
        data['dataset'],
        batch_size=data['batch_size'],
        shuffle=shuffle,
        #num_workers=data['num_workers'],
        num_workers=0,
        collate_fn=data['collate_fn']
        )
    return new_dataloader


#train_loader = unpickle_loader('train_loader')
# val_loader = unpickle_loader('val_loader')
test_loader = unpickle_loader('test_loader', shuffle=False)



print('Loading model')
model = UNet()

model_dir = os.path.join(current_dir, 'unet_params_ex')
state_dict_name = 'model_weights'  
state_dict_path = os.path.join(model_dir, '{}.pth'.format(state_dict_name))  
model.load_state_dict(torch.load(state_dict_path, weights_only=True, map_location=torch.device('cpu')))  
model.eval()
print('model loaded')




#%%

file_path = os.path.join(ex_data_dir, 'ordered_sliced_traces.hdf5')
# Open the HDF5 file
with h5py.File(file_path, 'r') as f:
    data_group = f['Data']

    t_load = data_group['TLoad'][:]
    # print("TLoad data:")
    # print(t_load)

    read_traces = {}
    for key in data_group.keys():
        if key.startswith('ReadTraces_'):
            time = int(key.split('_')[1])  
            read_traces[time] = data_group[key][:]
            # print(f"\nReadTraces at time {time}:")
            # print(read_traces[time])

t_L_array = []
n_blip_array = []

for t_L, traces in read_traces.items():
    t_L_array.append(t_L)

    with torch.no_grad():

        traces = np.array([test_loader.dataset.scale_transform(subtract_lb(trace)) for trace in traces])
        x = torch.tensor(traces, dtype=torch.float32).to(device)
        decoded_test_data = model(x)
        m = torch.nn.Softmax(dim=1)
        decoded_test_data = m(decoded_test_data)
        decoded_test_data = decoded_test_data.cpu().numpy()  # Get model output for visualization
        prediction_class = decoded_test_data.argmax(axis=1)
    
    n_blips = 0
    for trace in prediction_class:
        if np.min(trace) == 0:
            n_blips += 1
    n_blip_array.append(n_blips)        

n_blip_array = np.array(n_blip_array)
t_L_array = np.array(t_L_array)



#%%

ind = np.argsort(t_L_array)
t_L_array = t_L_array[ind]
n_blip_array = n_blip_array[ind]


def func1(x, A, G_in, T1, offset):
    return A*(np.exp(-x/T1) - np.exp(-G_in*x)) + offset
p01 = [200, 0.04, 90, 15]
popt1, pcov1 = sc.optimize.curve_fit(func1, t_L_array[t_L_array<200], n_blip_array[t_L_array<200])
print(popt1)

def func2(x, A, T1, offset): 
    return A*(np.exp(-x/T1)) + offset

popt2, pcov2 = sc.optimize.curve_fit(func2, t_L_array[t_L_array>100], n_blip_array[t_L_array>100])

def func3(x, A, G_in, T1, offset):
    return A*(1-np.exp((-G_in + 1/T1)*x)) + offset

popt3, pcov3 = sc.optimize.curve_fit(func1, t_L_array[t_L_array>100], n_blip_array[t_L_array>100])



prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
fig, ax = plt.subplots(1, 1)
ax.scatter(t_L_array, n_blip_array, marker='.', color = colors[1], alpha=0.6, label='Denoised experimental data')
#ax.plot(t_L_array,   np.exp(-lambda_flip * t_L_array), color='r', linestyle='dashdot')
# ax.plot(t_L_array[t_L_array<200], func1(t_L_array[t_L_array<200], *popt1), color = colors[1], label='$\sim \exp(-t_L/T_1)-\exp(-\Gamma_{in}t_L)$')
ax.plot(t_L_array, func1(t_L_array, *p01), color = 'red', label='$\sim \exp(-t_L/T_1)-\exp(-\Gamma_{in}t_L)$')

ax.plot(t_L_array, func3(t_L_array, *popt3), color = colors[0], linestyle=':', label='$\sim 1- \exp((-\Gamma_{in} + 1/T_{1})t_L)$')
ax.plot(t_L_array, func2(t_L_array, *popt2), color = colors[0], linestyle='dashdot', label='$\sim \exp(-t_L/T_1)$')
ax.set_xlabel(r'$t_L$ ($\mu$s)')
ax.set_ylabel(r'$N_{blip}$')
ax.legend()
plt.tight_layout()

fig_path = os.path.join(ex_data_dir, 'N_vs_tL.pdf')
plt.savefig(fig_path)





# #%%
# thresh_lower = 0.4
# thresh_upper = 0.6


# score = get_scores_unet(model, val_loader, start_read=0, end_read=-1, thresh_lower=thresh_lower, thresh_upper=thresh_upper)
# # score = get_scores_unet(model, val_loader, start_read=0, end_read=-1)
# print('score: ', score)

# #%%

# with torch.no_grad():  
#     x, y = next(iter(val_loader))
#     x = x.to(device)
#     y = y.to(device)
    
#     decoded_test_data = model(x)
#     m = torch.nn.Softmax(dim=1)
#     decoded_test_data = m(decoded_test_data)
#     decoded_test_data = decoded_test_data.cpu().numpy()  # Get model output for visualization
#     denoised_traces = schmitt_trigger(decoded_test_data[:, 1, :], full_output=False, thresh_lower=thresh_lower, thresh_upper=thresh_upper)
#     denoised_traces = denoised_traces + 1


#     for i in range(32):
#         fig, axs = plt.subplots(3, 1, figsize=(15, 5), sharex=True)  # Create a figure with 4 subplots
#         fig.suptitle('Test Trace')
#         axs[0].plot(x[i].reshape(-1), label='Noisy', color='mediumblue', linewidth=0.9)
#         axs[0].tick_params(labelbottom=False)
#         axs[1].plot(denoised_traces[i], label='Denoised', color='mediumblue', linewidth=0.9)
#         axs[1].tick_params(labelbottom=False)
#         axs[1].set_ylim(-0.1, 1.1)
#         axs[2].plot(decoded_test_data[i, 1, :], label='$p(1)$', color='mediumblue', linewidth=0.9)
#         axs[2].set_ylim(-0.1, 1.1)
#         axs[2].axhline(thresh_lower, color='red', linestyle='--')
#         axs[2].axhline(thresh_upper, color='red', linestyle='--')
        
#         for ax in axs:
#             ax.legend()
#             #ax.set_ylim(-0.1, 1.1)
#         #plt.show(block=False)
#         plt.savefig(os.path.join(ex_data_dir, f'unet_test_{i}.pdf'))  # Save each figure