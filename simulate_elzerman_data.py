#%%
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import gc
from numba import njit

#%%
# possible states: empty (0), spin up (1), spin down (-1)
# possible transitions: tunneling in (up and down), tunneling out (up and down)(, decay (up to down))

@njit
def get_t_next(lambda_current):
    u = np.random.uniform(0, 1)
    t_next_event = -np.log(u) / lambda_current
    return t_next_event

@njit
def get_lambda_in(E, spin, lambda_in, B, T):
    k_B = 8.617333262e-5 #eV/K
    delta_SO = 60e-6 #eV
    g_S = 2
    mu_B = 5.7883818060e-5 #eV·T−1. 
    delta_E = delta_SO + g_S *mu_B * B
    if spin == -1:
        E += delta_E
    return lambda_in * 1/(np.exp((E)/(k_B * T))+1)
@njit
def get_lambda_out(E, spin, lambda_out, B, T):
    k_B = 8.617333262e-5 #eV/K
    delta_SO = 60e-6 #eV
    g_S = 2
    mu_B = 5.7883818060e-5 #eV·T−1. 
    delta_E = delta_SO + g_S *mu_B * B
    
    if spin == -1:
        E += delta_E
    return lambda_out *1/(np.exp(-(E)/(k_B * T))+1)


@njit
def generate_elzerman_signal(lambdas, times, voltages, N_traces, n_samples_tot, signal_amp, T=9.0e-3, B=1.7):
    lambda_in, lambda_out, lambda_flip = lambdas
    t_L, t_W, t_R, t_U = times
    V_L, V_W, V_R, V_U = voltages

    t_rep = t_L + t_W + t_R + t_U
    n_samples = int(n_samples_tot / N_traces)
    dt = N_traces * t_rep/(n_samples_tot)
    continuous_states_total = np.zeros(n_samples_tot, dtype=np.float32)
    n_blip = 0

    state = 0
    for i in range(N_traces):
        time = 0.0
        state = 0
        times = np.array([0], dtype=np.float64)
        states = np.array([0], dtype=np.int64)

        # Load
        lambda_in_up = get_lambda_in(V_L, 1, lambda_in, B, T)
        lambda_in_down = get_lambda_in(V_L, -1, lambda_in, B, T)
        
        while time < t_L:
            if state == 0:
                t_up = get_t_next(lambda_in_up)
                t_down = get_t_next(lambda_in_down)
                if t_up < t_down and time + t_up < t_L:
                    t = t_up
                    state = 1
                elif t_up > t_down and time + t_down < t_L:
                    t = t_down
                    state = -1
                else:
                    time = t_L
                    t = 0.0
            elif state == -1:
                t_flip = get_t_next(lambda_flip)
                if time + t_flip < t_L:
                    t = t_flip
                    state = 1
                else:
                    time = t_L
                    t = 0.0
            else:
                time = t_L
                t = 0.0
            
            time += t
            times = np.append(times, time)
            states = np.append(states, state)
        
        time = t_L

        # Wait
        lambda_in_up = get_lambda_in(V_W, 1, lambda_in, B, T)
        lambda_in_down = get_lambda_in(V_W, -1, lambda_in, B, T)
        lambda_out_down = get_lambda_out(V_W, -1, lambda_out, B, T)
        lambda_out_up = get_lambda_out(V_W, 1, lambda_out, B, T)

        while time < t_L + t_W:
            if state == 0:
                t_up = get_t_next(lambda_in_up)
                t_down = get_t_next(lambda_in_down)
                if t_up < t_down and time + t_up < t_L + t_W:
                    t = t_up
                    state = 1
                elif t_up > t_down and time + t_down < t_L + t_W:
                    t = t_down
                    state = -1
                else:
                    time = t_L + t_W
                    t = 0.0
            elif state == -1:
                t_flip = get_t_next(lambda_flip)
                t_none = get_t_next(lambda_out_down)
                if t_flip < t_none and time + t_flip < t_L + t_W:
                    t = t_flip
                    state = 1
                elif t_flip > t_none and time + t_none < t_L + t_W:
                    t = t_none
                    state = 0
                else:
                    time = t_L + t_W
                    t = 0.0
            else:
                t_none = get_t_next(lambda_out_up)
                if time + t_none < t_L + t_W:
                    t = t_none
                    state = 0
                else:
                    time = t_L + t_W
                    t = 0.0
            
            time += t
            times = np.append(times, time)
            states = np.append(states, state)
        
        time = t_L + t_W

        # Read
        lambda_out_down = get_lambda_out(V_R, -1, lambda_out, B, T)
        lambda_in_up = get_lambda_in(V_R, 1, lambda_in, B, T)
        blip_time = [0.0, 0.0]
        
        while time < t_L + t_W + t_R:
            if state == 1:
                time = t_L + t_W + t_R
                t = 0.0
            elif state == -1:
                t_none = get_t_next(lambda_out_down)
                t_flip = get_t_next(lambda_flip)
                if t_none < t_flip and time + t_none < t_L + t_W + t_R:
                    t = t_none
                    state = 0
                    n_blip += 1
                    blip_time[0] = time + t
                elif t_none > t_flip and time + t_flip < t_L + t_W + t_R:
                    t = t_flip
                    state = 1
                else:
                    time = t_L + t_W + t_R
                    t = 0.0
            elif state == 0:
                t_up = get_t_next(lambda_in_up)
                if time + t_up < t_L + t_W + t_R:
                    t = t_up
                    state = 1
                    if blip_time != [0.0, 0.0]:
                        blip_time[1] = time + t
                        
                else:
                    time = t_L + t_W + t_R
                    t = 0.0
            
            time += t
            times = np.append(times, time)
            states = np.append(states, state)
        
        time = t_L + t_W + t_R

        # Unload
        lambda_out_up = get_lambda_out(V_U, 1, lambda_out, B, T)
        lambda_out_down = get_lambda_out(V_U, -1, lambda_out, B, T)

        while time < t_L + t_W + t_R + t_U:
            if state == 1:
                t_none = get_t_next(lambda_out_up)
                if time + t_none < t_L + t_W + t_R + t_U:
                    t = t_none
                    state = 0
                else:
                    time = t_L + t_W + t_R + t_U
                    t = 0.0
            elif state == -1:
                t_none = get_t_next(lambda_out_down)
                t_flip = get_t_next(lambda_flip)
                if t_none < t_flip and time + t_none < t_L + t_W + t_R + t_U:
                    t = t_none
                    state = 0
                elif t_none > t_flip and time + t_flip < t_L + t_W + t_R + t_U:
                    t = t_flip
                    state = 1
                else:
                    time = t_L + t_W + t_R + t_U
                    t = 0.0
            else:
                time = t_L + t_W + t_R + t_U
                t = 0.0
            
            time += t
            times = np.append(times, time)
            states = np.append(states, state)
        
        states = np.abs(states)
        continuous_states = np.zeros(n_samples, dtype=np.float32)
        for j in range(len(times) - 1):
            start_index = int(times[j] / dt)
            end_index = int(times[j+1] / dt)
            continuous_states[start_index:end_index] = states[j]
    
        #continuous_states = (continuous_states * 2 - 1) * signal_amp
        continuous_states_total[i * n_samples:(i + 1) * n_samples] = continuous_states
    
        blip_mask = np.zeros(n_samples, dtype=np.int64)
        for j in range(len(blip_time) - 1):
            start_index = int(blip_time[j] / dt)
            end_index = int(blip_time[j+1] / dt)
            blip_mask[start_index:end_index] = 1    
    return n_blip, blip_mask, continuous_states_total

@njit
def generate_dummy_trace(t_load, t_read, t_unload, amp=False, N=8192):
    
    times = np.array([t_load, t_read, t_unload])
    t_rep = np.sum(times)

    dt = t_rep/(8192)
    
    #times = times/t_rep * N
    t_load, t_read, t_unload = times
    
    t_in_rand = np.abs(np.random.triangular(0, 0, t_load))
    t_out_rand = np.random.triangular(t_load + t_read, t_load + t_read, t_rep)
    
    times = [0, t_in_rand, t_out_rand, t_rep]
    states = [0, 1, 0, 0]
    continuous_states = np.zeros(8192)
    for j in range(len(times) - 1):
        start_index = int(times[j] / dt)
        end_index = int(times[j+1] / dt)
        continuous_states[start_index:end_index] = states[j]
    
    continuous_states = (continuous_states * 2 - 1) * amp
    
    
    # #print(t_load_rand, t_in_rand)
    # states = np.zeros(N)
    # states[int(t_load_rand):int(t_in_rand)] = 1
    # states = (states * 2 - 1) * amp    
    
    return continuous_states



@njit()
def log_out(x, A, k):
    return A / (np.exp(-k*x) + 1)

@njit()
def log_in(x, A, k):
    return A / (np.exp(k*x) + 1)

@njit()
def sample_tunnel_rates(A0, k):
    A = np.abs(np.random.normal(A0, 1))
    sample_pos = np.random.normal(0.05, 0.075)
    lambda_in = log_in(sample_pos, A, k) * 1e3 # Tunneling in rate in Hz
    lambda_out = log_out(sample_pos, A, k) * 1e3  # Tunneling out rate in Hz
    print(lambda_in, lambda_out)
    return lambda_in, lambda_out

def noise(trace, sim_t, sigma, amps, freqs):
    
    shape = np.size(trace)
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
      
    total_noise = white_noise + pink_noise + interference_noise
    signal_power = np.mean(trace**2)
    noise_power = np.mean(total_noise**2)
    snr = signal_power/noise_power
    print(signal_power)
    print(noise_power)
    snr = 10*np.log10(snr)

    noisy_trace = trace + total_noise
    return noisy_trace, snr

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def main():
    # Parameters for Elzerman
    B = 2.1 
    delta_SO = 60e-6 #eV
    g_S = 2
    mu_B = 5.7883818060e-5 #eV·T−1. 
    delta_E = delta_SO + g_S *mu_B * B
    
    V_L = -9e-3 #eV
    V_W = -delta_E #eV
    V_R = -delta_E/2 #eV
    V_U = 7.5e-3 #eV
    voltages = [V_L, V_W, V_R, V_U]


    lambda_flip = 1/(8.37e-3) # min 20

    t_L = 0.5e-3
    t_W = 0.0
    t_R = 1.0e-3
    t_U = 1.5e-3
    
    lambda_in = 4200.0
    lambda_out = 3500.0
    n_samples = 15000

    s=0.01
    noise_std = s  # Standard deviation of Gaussian noise
    T = t_L + t_W + t_R + t_U  # Total simulation time in seconds
    print(f'Simulation time per trace: {T}s\n\n')

    interference_amps = [s, s, s, s]  # Amplitudes of the interference signals
    interference_freqs = [50, 200, 600, 1000]  # Frequencies of the interference signals in Hz

    
    signal_amp = 1.0

    n_samples_tot = 15000
    _, mask, trace = generate_elzerman_signal([lambda_in, lambda_out, lambda_flip], [t_L, t_W, t_R, t_U], voltages, 1,n_samples_tot, signal_amp)
    trace, snr = noise(trace, T, s, interference_amps, interference_freqs)
    print(f'Signal to noise: {snr} dB')
    times = np.arange(0, n_samples_tot, 1)
    fig, (ax, bx) = plt.subplots(2, 1)
    ax.plot(trace, label='Simulated test data')
    ax.plot(times[mask==1], trace[mask==1], color='red', linewidth=5, alpha=0.5, label='Actual anomalies')
    ax.legend()
    ax.set_title('Preview Trace')
    bx.hist(trace, bins=150)
    plt.show(block=True) 


    def save_dummy_traces(file_name, n): 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        trace_dir = os.path.join(current_dir, 'sim_traces')
        hdf5_file_path = os.path.join(trace_dir, '{}.hdf5'.format(file_name))
        
        with h5py.File(hdf5_file_path, 'w') as file:
            start_time = time.perf_counter()
            for i in range(n):
            
                data = generate_dummy_trace(t_L + t_W, t_R, t_U, signal_amp)
                name = f'trace_{i}'
                if name in file:
                    del file[name]  # Delete the existing dataset
                file.create_dataset(name, data=data)
                #print(name)
                # Explicitly delete variables and force garbage collection
                del data
                gc.collect()
                remaining_time = round((time.perf_counter() - start_time)/(i+1) * (n-i-1), 2)
                printProgressBar(i+1, n, f'\rSimulating dummy traces... Time remaining: {remaining_time:.2f}s', 'complete', length=25)
                
            end_time = time.perf_counter()
            print('...took {}s\n'.format((end_time - start_time)))

    def save_elzerman_traces(file_name, n):
         current_dir = os.path.dirname(os.path.abspath(__file__))
         trace_dir = os.path.join(current_dir, 'sim_traces')
         os.makedirs(trace_dir, exist_ok=True)  
         hdf5_file_path = os.path.join(trace_dir, '{}.hdf5'.format(file_name))

         with h5py.File(hdf5_file_path, 'w') as file:
            start_time = time.perf_counter()
            for i in range(n):
            
                _, _, data = generate_elzerman_signal([lambda_in, lambda_out, lambda_flip], [t_L, t_W, t_R, t_U], voltages, 1, 8192, signal_amp)
                name = f'trace_{i}'
                if name in file:
                    del file[name]  # Delete the existing dataset
                file.create_dataset(name, data=data)
                #print(name)
                # Explicitly delete variables and force garbage collection
                del data
                gc.collect()
                remaining_time = round((time.perf_counter() - start_time)/(i+1) * (n-i-1), 2)
                
                printProgressBar(i+1, n, f'Simulating Elzerman traces... Time reamaining: {remaining_time:.2f}s', 'complete', length=25)
                
            end_time = time.perf_counter()
            print('...took {}s\n'.format((end_time - start_time)))

    def save_elzerman_traces_and_masks(file_name, hdf5_file_path_masks, n):
         current_dir = os.path.dirname(os.path.abspath(__file__))
         trace_dir = os.path.join(current_dir, 'sim_traces')
         hdf5_file_path = os.path.join(trace_dir, '{}.hdf5'.format(file_name))

         mask_file = h5py.File(hdf5_file_path_masks, 'w')
         with h5py.File(hdf5_file_path, 'w') as file:
            start_time = time.perf_counter()
            for i in range(n):
                

                _, mask, data = generate_elzerman_signal([lambda_in, lambda_out, lambda_flip], [t_L, t_W, t_R, t_U], voltages, 1, n_samples, signal_amp)
                name = f'trace_{i}'
                if name in file:
                    del file[name]  # Delete the existing dataset
                file.create_dataset(name, data=data)
                if name in mask_file:
                    del mask_file[name]  # Delete the existing dataset
                mask_file.create_dataset(name, data=mask)
                
                # Explicitly delete variables and force garbage collection
                del data, mask
                gc.collect()
                remaining_time = round((time.perf_counter() - start_time)/(i+1) * (n-i-1), 2)
                
                printProgressBar(i+1, n, f'Simulating Elzerman traces... Time reamaining: {remaining_time:.2f}s', 'complete', length=25)
                
            end_time = time.perf_counter()
            print('...took {}s\n'.format((end_time - start_time)))

    def save_read_traces(file_name, n):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        trace_dir = os.path.join(current_dir, 'sim_traces')
        os.makedirs(trace_dir, exist_ok=True)  
        hdf5_file_path = os.path.join(trace_dir, '{}.hdf5'.format(file_name))

        with h5py.File(hdf5_file_path, 'w') as file:
            start_time = time.perf_counter()
            
            times = np.array([t_L, t_L + t_W, t_L + t_W + t_R, t_L + t_W + t_R + t_U])
            
            times_indices = times * n_samples / T
            times_indices = times_indices.astype(np.int64)
            start_read, end_read = times_indices[1], times_indices[2]
            # if (end_read - start_read)%2 == 1:
            #     end_read = end_read - 1
            print('Length of read trace: ', end_read - start_read)
            for i in range(n):
                #lambda_in , lambda_out = np.exp(np.random.uniform(np.log(1000), np.log(20000), 2))
                #print(lambda_in, ' ', lambda_out)
                _, _, data = generate_elzerman_signal([lambda_in, lambda_out, lambda_flip], [t_L, t_W, t_R, t_U], voltages, 1, n_samples, signal_amp)
                data = data[start_read:end_read]
                name = f'trace_{i}'
                if name in file:
                    del file[name]  # Delete the existing dataset
                file.create_dataset(name, data=data)
                #print(name)
                # Explicitly delete variables and force garbage collection
                del data
                gc.collect()
                remaining_time = round((time.perf_counter() - start_time)/(i+1) * (n-i-1), 2)
                if i%100 == 0:
                    printProgressBar(i+1, n, f'Simulating Elzerman traces... Time reamaining: {remaining_time:.2f}s', 'complete', length=25)
                
            end_time = time.perf_counter()
            print('...took {}s\n'.format((end_time - start_time)))


    #save_read_traces('sim_read_traces_train_10k_pure', 10000)
    #save_read_traces('sim_read_traces_val_mixed', 100)
    # save_read_traces('sim_read_traces_train_10k', 10000)
    # save_read_traces('sim_read_traces_val', 100)
    
    #save_read_traces('sim_read_traces_train_20k_pure', 20000)
    
    #save_read_traces('sim_elzerman_traces_train_1k', 1000)
    
    save_elzerman_traces('sim_elzerman_traces_train_1k', 1000)
    save_elzerman_traces('sim_elzerman_traces_train_10k', 10000)
    
    save_elzerman_traces('sim_elzerman_traces_test_1k', 1000)
    save_elzerman_traces('sim_elzerman_traces_test_10k', 10000)
    
    save_elzerman_traces('sim_elzerman_traces_val', 100)
    
    

if __name__ == '__main__':
    main()
#%%

import h5py
import numpy as np
import matplotlib.pyplot as plt

def show_data(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.path.join(current_dir, 'sim_traces')
    hdf5_file_path = os.path.join(current_dir, '{}.hdf5'.format(file_name))
    with h5py.File(hdf5_file_path, 'r') as h5f:
        all_keys = h5f.keys() 
    
        for key in all_keys:    
            data = np.array([h5f[key]])
            print(data)
            fig, ax = plt.subplots(1, 1)
            ax.plot(data[0])
            
            plt.show(block=True)
        