#%%
import numpy as np 
import matplotlib.pyplot as plt
import os
import h5py 
from HDF5Data import HDF5Data

current_dir = os.path.dirname(os.path.abspath(__file__))
trace_dir = os.path.join(current_dir, 'sim_traces')
real_data_dir = os.path.join(current_dir, 'real_data')
test_trace_name = "494_2_3T_elzermann_testtrace_at_b'repitition'_771.000_b'Pulse_for_Qdac - Tburst'_554.500"
test_name = 'sliced_traces' 
hdf5_file_path_test = os.path.join(real_data_dir, '{}.hdf5'.format(test_name))  
test_trace_path = os.path.join(real_data_dir, '{}.npy'.format(test_trace_name))  

noise_name = '545_1.9T_pg13_vs_tc'
noise_path = os.path.join(real_data_dir, '{}.hdf5'.format(noise_name))

test_trace = np.load(test_trace_path)
test_trace = test_trace[:15000]
test_trace = test_trace[-5000:]

hdf5Data_noise = HDF5Data(wdir=trace_dir)
hdf5Data_noise.set_path(noise_path)

hdf5Data_noise.set_filename()
hdf5Data_noise.set_traces()
hdf5Data_noise.set_measure_data_and_axis()
noise_traces = np.array(hdf5Data_noise.traces) #self.data.traces
tc = np.array(hdf5Data_noise.measure_axis[1]).T

mask = np.logical_and(8e-6<tc, tc<12e-6)
noise_traces = noise_traces[mask]

for trace in noise_traces:
    fig, ax = plt.subplots(1, 1)
    start = np.random.randint(0, len(trace)-5000)
    stop = start + 5000
    noise = trace[start:stop]
    noise -= np.mean(noise)
    noise /= np.max(np.abs(noise))

    ax.plot(noise)
    #ax.set_ylim(0.8, 2)
    plt.show(block=True)



plt.plot(test_trace)