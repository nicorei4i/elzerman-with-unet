#%%

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from simulate_elzerman_data import generate_elzerman_signal 
from ElzerData import ElzerData
import time
from scipy.signal import welch
plt.switch_backend('TkAgg') 

#noise_path = '//serveri2a/Transfer/Nico/01.Data/Elzerman/545_1.9T_pg13_vs_tc.hdf5'
# real_trace_path = '//serveri2a/Transfer/Nico/01.Data/Elzerman/494_2_3T_elzermann_test.hdf5'
real_trace_path = '//serveri2a/Transfer/Nico/01.Data/Elzerman/492_2_5T_elzermann_ - Copy.hdf5'


current_dir = os.path.dirname(os.path.abspath(__file__))
trace_dir = os.path.join(current_dir, 'sim_traces')

tick = time.perf_counter()
print('Loading data...')
script_dir = os.path.dirname(os.path.abspath(__file__))
# if not os.path.exists(trace_dir):
#     os.makedirs(wdir=trace_dir)
#hdf5Data_noise = ElzerData(wdir=trace_dir)
#hdf5Data_noise.set_path(noise_path)

#hdf5Data_noise.set_filename()
#hdf5Data_noise.set_traces()

#hdf5Data_noise.set_traces_dt() # self.data.set_traces_dt()
#noise_dt = hdf5Data_noise.traces_dt
#noise_traces = hdf5Data_noise.traces #self.data.traces
#noise_time_axis = noise_dt * np.arange(0, len(noise_traces[0][0]))
#print('Noise traces: ', noise_traces.shape)  


hdf5Data_traces = ElzerData(wdir=trace_dir, t_ini=1000, t_read=1000,  sampling_rate=2)
hdf5Data_traces.set_path(real_trace_path)
hdf5Data_traces.set_filename()
hdf5Data_traces.set_traces()

# real_traces = hdf5Data_traces.traces
# fig, ax = plt.subplots(1, 1)
# ax.plot(real_traces[46][78])


hdf5Data_traces.set_traces_dt() # self.data.set_traces_dt()
# traces_dt = hdf5Data_traces.traces_dt
# traces = hdf5Data_traces.traces #self.data.traces
# print('Measured traces: ', traces.shape)  

file_name = 'sliced_traces_ordered'  
hdf5Data_traces.create_new_file_only_traces(file_name)
#hdf5Data_traces.cut_traces()

hdf5_file_path = os.path.join(trace_dir, '{}.hdf5'.format(file_name))   

# with h5py.File(hdf5_file_path, 'r') as file:  
#     all_keys = file.keys()  
#     sliced_traces = np.array([file[key] for key in all_keys],dtype=np.float32)  
#     print('Sliced measured traces: ', sliced_traces.shape)  
sliced_traces = np.array(hdf5Data_traces.read_traces)
# whole_traces = np.array(hdf5Data_traces.padded_arrays)
print('Sliced measured traces: ', sliced_traces.shape)  
# print('Sliced whole measured traces: ', whole_traces.shape)  


#traces_time_axis = traces_dt * np.arange(0, len(noise_traces[0][0]))


# file_name = 'sim_read_traces_train'  
# hdf5_file_path = os.path.join(trace_dir, '{}.hdf5'.format(file_name))   

# with h5py.File(hdf5_file_path, 'r') as file:  
#     all_keys = file.keys()  
#     sim_traces = np.array([file[key] for key in all_keys], dtype=np.float32)  
#     print('Simulated read traces: ', sim_traces.shape)  

# for i, trace in enumerate(whole_traces):
#     fig, ax = plt.subplots(1, 1)
#     ax.plot(trace)
#     fig.suptitle(f'Trace {i}') 
#     mng = plt.get_current_fig_manager()
#     mng.window.state('zoomed')

#     plt.show(block=True)
 
# #%%
# for i_sim in np.random.randint(0, len(sim_traces), 10): 
#     i_noise = np.random.randint(noise_traces.shape[0])
#     j_noise = np.random.randint(noise_traces.shape[1])
    
#     i_real = np.random.randint(sliced_traces.shape[0])
    
#     sim_trace = sim_traces[i_sim]
#     noise_trace = noise_traces[i_noise, j_noise]
#     real_trace = sliced_traces[i_real]
    
#     start = np.random.randint(0, len(noise_trace)-len(sim_trace))
#     stop = start + len(sim_trace)
#     noise = noise_trace[start:stop]
#     noise -= np.mean(noise)
#     sim_noise_trace = sim_trace + 20*noise



    
#     fig, ax = plt.subplots(3, 1, figsize=(12, 6))
#     ax[0].plot(sim_trace)
#     ax[1].plot(sim_noise_trace)
#     ax[2].plot(real_trace)
# #    ax[3].plot(whole_traces[i_real])
#     plt.show(block=True)

