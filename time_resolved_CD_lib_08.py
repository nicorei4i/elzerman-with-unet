# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:20:58 2023

@author: Valerius
"""

# -*- coding: utf-8 -*-

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from matplotlib import rc
from scipy.signal import butter, lfilter
import time
from numba import jit


plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cmr10"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

# LaTeX fonts
# rc('pdf', fonttype = 42)
# rc('text', usetex = True)
# rc('font', family = 'serif')

# +++++++++++++++++++++++++++++
# Mathematical functions
# +++++++++++++++++++++++++++++

def gaussian(x, a, m, s):
    y = a * np.exp(-((x - m) / s) ** 2 / 2)
    return y

def double_gaussian(x, a1, m1, s1, a2, m2, s2, offset):
    y = a1 * np.exp(-((x - m1) / s1) ** 2 / 2) + a2 * np.exp(-((x - m2) / s2) ** 2 / 2) + offset
    return y

def fermi(E, A, E_F, kT):
    y =  A / (1 + np.exp((E - E_F) / kT))
    return y

def exponential(t,gamma_A, gamma_B, b):
    y = gamma_A * np.exp(-gamma_B*t) + b
    return y    

def lorentzian(x,A,x0,width):
    y = A / (1 + ((x - x0) / (width / 2))**2)
    return y

def cosh_function(x, G, x0, kB_T):
    y = G * 1/(np.cosh((x-x0)/kB_T))**2
    return y

def quadratic_mean(data):
    return np.sqrt(np.mean(np.square(data)))

def probability_tau_hist(t, G_up, G_down, a):
    y_1 = (np.exp(-G_up*t)- np.exp(-G_down*t))/(1/G_up - a/G_down)
    y_2 = a*(np.exp(-G_up*t)- np.exp(-G_down*t))
    
    return y_1

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                            Fitting functions
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    
def fit_double_gaussian(x_data, y_data, p0, bounds):
    try:
        params, cov = curve_fit(double_gaussian, x_data, y_data, p0, bounds = bounds)
        
    except RuntimeError:
        params, cov = np.zeros(7), 0

    return params, cov

def fit_gauss_hist(hist, bin_edges, p0_l, p0_u):
    bin_centers = bin_edges #0.5*(bin_edges[1:] + bin_edges[:-1])
    mp = len(bin_centers) // 2
    
    var = 0 
    
    try: 
        params1, covariance1 = curve_fit(gaussian, bin_centers[:mp], hist[:mp], p0 = p0_l)#p0=[30000, -0.013, 0.001]) #p0=[10300, -0.013, 0.001]) 
        params2, covariance2 = curve_fit(gaussian, bin_centers[mp:], hist[mp:], p0=p0_u) #p0=[30000, -0.013, 0.001]) #p0=[12000, -0.0094, 0.001]) 
    
    except RuntimeError as e:
        var = 1 
        params1 = 0
        params2 = 0 

    fit_x1 = np.linspace(min(bin_centers[:mp]), max(bin_centers[:mp]), 500)
    fit_y1 = gaussian(fit_x1, *params1)
    #plt.plot(fit_x1, fit_y1, color='r')

    fit_x2 = np.linspace(min(bin_centers[mp:]), max(bin_centers[mp:]), 500)
    fit_y2 = gaussian(fit_x2, *params2)
    
    fit_data = [fit_x1, fit_y1, fit_x2, fit_y2]
    #plt.plot(fit_x2, fit_y2, color='b')    
    return params1, params2, var, fit_data
     
def fit_fermi(x_data, y_data, params_guess):
    p0 = params_guess
    params, cov = curve_fit(fermi, x_data, y_data, p0)
    return params, cov


# +++++++++++++++++++++++++++++
# Functions for data processing
# +++++++++++++++++++++++++++++
def read_file_no_time(file_path, group, dataset, channels, information): # read function for hdf5 files with no time dimension
    with h5py.File(file_path, 'r') as file:
        channel_data_lists = {}  # Creating an empty dictionary for measurement data
        group_name = group 
        dataset_name = dataset  
        channel_names = channels
        
        if information:
            def print_name(name):
                print(name)

            # Use the visit method to traverse all groups and datasets
            print("Groups and datasets in the HDF5 file:")
            file.visit(print_name)

            # List all datasets in the file
            dataset_names = list(file.keys())
            print("Datasets in the HDF5 file:", dataset_names)
    
            # Print attributes of the group
            for attr_name, attr_value in file[group].attrs.items():
                print(f"Attribute: {attr_name} = {attr_value}")
            
        else:
            if group_name in file:
                group = file[group_name]
                data_dataset = file[dataset_name]
                channel_names_dataset = file[channel_names]
                
                # Initialize an empty list to store channel names
                channel_names = []

                # Iterate through the dataset and access the elements of each tuple
                for item in channel_names_dataset:
                    channel_name = item[0]  # Access the first element of the tuple
                    channel_names.append(channel_name)

                channel_names = [channel_name.decode('utf-8') for channel_name in channel_names]

                for name in channel_names:
                    print(name)

                # Convert the dataset to a NumPy array
                data = data_dataset[:]

                # Create lists with channel names containing the corresponding data
                for i, channel_name in enumerate(channel_names):
                    channel_data = data[:, i, :]  # Extract data for the current channel
                    channel_data_lists[channel_name] = channel_data.tolist()

                # Dictionary where keys are channel names, and values are lists of corresponding data
                for channel_name, channel_data in channel_data_lists.items():
                    print(f"Channel Name: {channel_name}")
                    print(np.shape(channel_data))
                    print()
                    
    return file, channel_data_lists


def read_time(file_path, group, dataset, time_spacing, n_points):
    
    with h5py.File(file_path, 'r') as file:
        
        group_name = group 
        dataset_name = dataset  
        time_spacing_name = time_spacing # array with t0dt
        
        if group_name in file:
            group = file[group_name]
                
        if dataset_name in file and time_spacing_name in file: 
            data = file[dataset_name][:]
            time_spacing = file[time_spacing_name][:]
            
            time_array = np.linspace(0, time_spacing[0][1]*n_points, n_points)
        
        else: 
            print("Dataset not found")
        
    return time_array

def read_file(file_path, group, dataset, time_spacing, information): 
    """
    information: True/False, either reads file or prints information containing data structure
    
    returns array with time and array containing traces with ydata
    """
    
    verbose = True
    
    start_time = time.time()
    with h5py.File(file_path, 'r') as file:
        
        group_name = group 
        dataset_name = dataset  
        time_spacing_name = time_spacing # array with t0dt
        
        if group_name in file:
            group = file[group_name]
                
        if dataset_name in file and time_spacing_name in file: 
            data = file[dataset_name][:]
            time_spacing = file[time_spacing_name][:]
            
            n_traces = len(data[0][0])   
        
        end_time = time.time()
        read_time = end_time - start_time
        
        if verbose:
            print(f"Reading time: {read_time:.2f} seconds.")
        
        if information:
            def print_name(name):
                print(name)

            # Use the visit method to traverse all groups and datasets
            print("Groups and datasets in the HDF5 file:")
            file.visit(print_name)

            # List all datasets in the file
            dataset_names = list(file.keys())
            print("Datasets in the HDF5 file:", dataset_names)
    
            # Print attributes of the group
            for attr_name, attr_value in group.attrs.items():
                print(f"Attribute: {attr_name} = {attr_value}")
        
            time_array, traces_array = [], []
            
            print("t0 - dt:")
            print(file['Traces/Alazar Slytherin - Ch1 - Data_t0dt'][0])
            
        else:
            
            start_time_np = time.time()
            
            time_spacing_list = []
            traces_list = []
                 
            for i in range(n_traces):
                traces_list.append([]) # adding n_traces empty lists to array, which will be populated with data
                #time_spacing_list.append(time_spacing[i][1]) # Sometimes time spacing is the same for all traces, then i=0 for all traces
                time_spacing_list.append(time_spacing[0][1]) 
                       
            # Extract different traces from data
            for i in data:
                val = i[0]
                for j in range(n_traces):
                    t_val = val[j]
                    if np.isnan(t_val) == False: 
                        traces_list[j].append(t_val)
                    else: 
                        continue
            
            traces_array = np.array(traces_list)

            # Create time array according to determined spacing
            time_array = [np.linspace(0, t*len(traces_array[i]), len(traces_array[i])) for t,i in zip(time_spacing_list, range(n_traces))]
            end_time_np = time.time()
            np_array_time = end_time_np - start_time_np
            
            if verbose:
                print(f"Numpy array creation time: {np_array_time:.2f} seconds")
            
        
    return time_array, traces_array
    

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#        Functions for plotting time traces, either smoothed or normalized 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                      
def plot_time_trace(n, time, traces_array, xmin, xmax, smoothing, normalize):
    #plt.figure()
    #x_data = time[n]
    x_data = time
    filter_params = [30,1000,0.0005,5,300]
    if smoothing == "average": 
        window = filter_params[0]
        y_data = moving_average(traces_array[n], window)[window:-window]
        #x_data = time[n][window:-window]
        x_data = time[window:-window]
        
    elif smoothing == "lowpass":
        warmup_points = filter_params[1]
        period = filter_params[2]    
        order = filter_params[3]
        y_mean_pts = filter_params[4] 
        
        fs = 1/(x_data[1] - x_data[0])
        cutoff_frequency = min(0.5 * fs, 1.0 / (2 * period))
        cutoff_frequency = min(0.5 * fs, cutoff_frequency)
        
        line_trace_data_adjusted = np.concatenate((traces_array[n][:warmup_points], traces_array[n]))
        
        filtered_data = butter_lowpass_filter(line_trace_data_adjusted, cutoff_frequency, fs, order)
        
        line_trace_data_plot = filtered_data[warmup_points:]
        y_data = line_trace_data_plot
        
    elif smoothing == None:
        y_data = traces_array[n]
            
    if normalize == True: 
        
        normalized_signal = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))
        y_data = normalized_signal
        
        # split data in upper and lower half
        
        mp = (np.max(y_data) - np.min(y_data))/2
                
        upper_half = [x for x in y_data if x > mp]
        lower_half = [x for x in y_data if x <= mp]
        
        upper_mean = np.mean(upper_half)
        lower_mean = np.mean(lower_half)

        
        normalized_list = [(x - lower_mean) / (upper_mean - lower_mean) for x in y_data]
        
        return x_data, normalized_list  
    
    elif normalize == False: 
        return x_data, y_data

def average_smooth_trace(n, time, traces_array):
    window = 30
    x_data = time
    y_data = traces_array[n]
    y_data = moving_average(y_data, window)[window:-window]
    x_data = x_data[window:-window]
    
    normalized_signal = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))
    y_data = normalized_signal
    
    return x_data, y_data 
    

def plot_bool_time_trace(y_data, threshold):
    input_array = y_data
    output_array = (input_array >= threshold).astype(int)
    return output_array

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                   Functions for making histograms of data 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def hist_data(x_data, y_data):
    hist, bin_edges, _ = plt.hist(y_data, bins = 150, density = False, color ="grey")
    hist, bin_edges = np.histogram(y_data, bins = 150, density = False)
    return hist, bin_edges

def hist_2d(FG_data, data, n_FG, idx_range):
    # Create array to store histogram results
    histograms = np.zeros((n_FG, 200))  # Assuming 30 bins for each histogram

    # Compute histograms for each time trace
    for i in range(n_FG):
        histograms[i], bin_edges = np.histogram(data[i], bins=200, density=True)  # Using density=True for normalized histograms

    # Plot the 2D histogram array
    plt.figure(figsize=(10, 8))
    plt.imshow(histograms.T, aspect='auto', cmap='viridis', origin='lower', extent=[min(FG_data), max(FG_data), min(bin_edges), max(bin_edges)])
    plt.axvline(x=FG_data[idx_range[0]], color = "r")
    plt.axvline(x=FG_data[idx_range[1]], color = "r")
    plt.colorbar(label='Normalized Count')
    plt.xlabel(r'$V_\mathrm{FG}$ [V]')
    plt.ylabel('Detector signal [mV]')
    plt.show()
    
def hist1d_array(data, idx_range):
    data_corr = data[idx_range[0]:idx_range[1]]
    data_1d = data_corr.flatten()
    # Create histogram of the flattened data
    hist, bins = np.histogram(data_1d, bins=200)
    bin_centers = 0.5*(bins[1:] + bins[:-1])

    return bin_centers, bins, hist
  
    
# +++++++++++++++++++++++++++++
# Smoothing tools
# +++++++++++++++++++++++++++++

def moving_average(data, window_size):
    half_window = window_size // 2
    smoothed_data = np.zeros_like(data)
    for i in range(half_window, len(data) - half_window):
        smoothed_data[i] = np.mean(data[i - half_window: i + half_window + 1])
    return smoothed_data

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y  
    
           
# +++++++++++++++++++++++++++++
# Determination of tunneling rates
# +++++++++++++++++++++++++++++

def gamma(t_list):
    
    try: 
        t_mean = np.mean(t_list)
        t_s = np.std(t_list)
        gamma = 1/t_mean
        gamma_s = gamma * t_s/t_mean
        
    except (TypeError, ZeroDivisionError, ValueError):
        print("Error occured")
        gamma = float("NaN")
        gamma_s = float("NaN")
    
    return gamma, gamma_s

def make_FG_array(n,FG_min,FG_max):
    FG_array = np.linspace(FG_min, FG_max, n)
    return FG_array  

def fit_gamma(x_data,y_data):
    params, cov = curve_fit(exponential, x_data, y_data)
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = exponential(x_fit, *params)
    return x_fit, y_fit, params

def find_av_tunneling(gamma_1, gamma_2):
    diff_lists = abs(np.array(gamma_1) - np.array(gamma_2))
    idx = np.argmin(diff_lists)
    
    gamma = (gamma_1[idx] + gamma_2[idx])/2
    
    return idx, gamma
          

def snr_calc(params): # needs fit parameters of double gaussian fit with format [a0,x0,s0,a1,x1,s1,offset]
    x0 = params[1]
    x1 = params[4]
    s0 = params[2]
    s1 = params[5]    
    d = x1 - x0 
    s_max = max(s0,s1)    
    snr = abs(d/s_max)
    return snr

def det_a(snr, m, b): 
    # determines parameter a used for Schmitt threshold
    a = m * snr + b
    return a



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#       Functions for detection algorithm, finds thresholds and blibse
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def gauss_params_bla(traces_array,n, offset, width_start, bounds_gaussian, bounds_double_gaussian): # n: trace number, offset: fit parameter for double gauss, width_start: gauss width start parameter, bounds: fit bounds for single and db Gauss
    # used to find threshold values. Fits double Gaussian to data and returns fit params    
    # this function needs histogram data of a trace, including bin centers and histogram values
    # it then checks whether a single Gauss or double Gauss is fitted and returns fit parameters

    hist, bins = np.histogram(traces_array[n], bins = 150, density = False)
    bin_centers = 0.5*(bins[1:] + bins[:-1])

    hist_smoothed = moving_average(hist, 20) # smooth histogram data, might have to be adjusted depending on nb of bins
    hist_diff = np.diff(hist_smoothed)

    # find peaks in histogram via peak finder, parameters might have to be adjusted
    peaks, _ = find_peaks(hist_smoothed, height = 100, distance = 30, prominence = 200)
    print(len(peaks))
    
    if len(peaks) == 1: # if only one peak is found, it's either one Gaussian or two very close to each other, which we will check now
        diff_abs = abs(hist_diff)
        peaks_diff, _ = find_peaks(diff_abs, height = 100, distance = 20, prominence = 50, width = 10) #adjust values
    
        if len(peaks_diff) == 2: # if only one peak, abs(diff) will have two maxima
            snr = 0 # no snr can be calculated since only one peak
            x0 = bin_centers[peaks][0] # set start parameters for single Gauss fit
            a0 = hist_smoothed[peaks][0]    
            p0 = [a0, x0, width_start]
            params, cov = curve_fit(gaussian, bin_centers, hist, p0, bounds = bounds_gaussian)
        
        elif len(peaks_diff) == 3: # if two peaks and above optical resolution, abs(diff) will have three maxima due to shoulder
            x0_idx = int((peaks_diff[1] + peaks_diff[0])/2) # peak in histogram is approximately located between peaks in diff
            x1_idx = int((peaks_diff[2] + peaks_diff[1])/2)
            
            x0 = bin_centers[x0_idx] # set start parameters for double gaussian fit
            x1 = bin_centers[x1_idx]
            
            a0 = hist[x0_idx]
            a1 = hist[x1_idx]
            
            p0 = [a0, x0, width_start, a1, x1, width_start, offset]

            params, cov = fit_double_gaussian(bin_centers, hist, p0, bounds_double_gaussian) 
            snr = snr_calc(params) # calculate snr

    elif len(peaks) == 2: # two distinct peaks in hist data
        x0 = bin_centers[peaks][0] # set start parameters for double gaussian fit 
        a0 = hist[peaks][0]       
        x1 = bin_centers[peaks][1]
        a1 = hist[peaks][1]     
        p0 = [a0, x0, width_start, a1, x1, width_start, offset]
        
        print(p0)
    
        params, cov = fit_double_gaussian(bin_centers, hist, p0, bounds_double_gaussian)     # replace with curve fit 
        snr = snr_calc(params)
    else: 
        print(len(peaks))
        params, snr = 0,0
        


    return bin_centers, hist, params, snr


def gauss_trace_normalized(trace, offset, width_start, bounds_gaussian, bounds_double_gaussian, n_bins):
    print("This function has been calles successfully")
    hist, bins = np.histogram(trace, bins = n_bins, density = False)
    bin_centers = 0.5*(bins[1:] + bins[:-1])

    hist_smoothed = moving_average(hist, 5) # smooth histogram data, might have to be adjusted depending on nb of bins
    # plt.figure()
    # plt.plot(bin_centers,hist_smoothed)

    # find peaks in histogram via peak finder, parameters might have to be adjusted
    peaks, _ = find_peaks(hist_smoothed, height = 100, distance = 30, prominence = 200)
    print(peaks)
    
    if len(peaks) == 1: # if only one peak is found, it's either one Gaussian or two very close to each other, which we will check now
        hist_diff = np.diff(hist_smoothed)   
        
        diff_abs = abs(hist_diff)
        plt.figure()
        plt.plot(bin_centers[:-1],diff_abs)
        peaks_diff, _ = find_peaks(diff_abs, height = 500, threshold = 5, distance = 10, prominence = 30, width = 10) #adjust values
        print("Peaks diff")
        print(peaks_diff)
        print(len(peaks_diff))
        
        if len(peaks_diff) == 2: # if only one peak, abs(diff) will have two maxima
            snr = 0 # no snr can be calculated since only one peak
            x0 = bin_centers[peaks][0] # set start parameters for single Gauss fit
            a0 = hist_smoothed[peaks][0]    
            p0 = [a0, x0, width_start]
            params, cov = curve_fit(gaussian, bin_centers, hist, p0, bounds = bounds_gaussian)
            
            
            return trace, bin_centers, hist, params, 0
            
        elif len(peaks_diff) == 3: # if two peaks and above optical resolution, abs(diff) will have three maxima due to shoulder
            x0_idx = int((peaks_diff[1] + peaks_diff[0])/2) # peak in histogram is approximately located between peaks in diff
            x1_idx = int((peaks_diff[2] + peaks_diff[1])/2)
            
            x0 = bin_centers[x0_idx] # set start parameters for double gaussian fit
            x1 = bin_centers[x1_idx]
            
            a0 = hist[x0_idx]
            a1 = hist[x1_idx]
            
            plt.axvline(x0)
            plt.axvline(x1)
            
            p0 = [a0, x0, width_start, a1, x1, width_start, offset]

            params, cov = fit_double_gaussian(bin_centers, hist, p0, bounds_double_gaussian) 
            snr = snr_calc(params) # calculate snr
            x_l = min(params[1],params[4])
            x_u = max(params[1],params[4])
            scaling_factor = 1/abs(params[1]-params[4])
            trace_norm = (trace-x_l)/(x_u-x_l)
            hist_norm, bins_norm = np.histogram(trace_norm, bins = n_bins, density = False)
            bin_centers_norm = 0.5*(bins_norm[1:] + bins_norm[:-1])
            
            p0 = [a0, 0, width_start*scaling_factor, a1, 1, width_start*scaling_factor, offset]
            
            bounds_double_gaussian_norm = ([0,-0.2,0,0,0,0,0],[1e6,0.5,0.5,1e6,1.2,0.5,100])
            
            params_norm, cov_norm = fit_double_gaussian(bin_centers_norm, hist_norm, p0, bounds_double_gaussian_norm)
            
            return trace_norm, bin_centers_norm, hist_norm, params_norm, snr

    elif len(peaks) == 2: # two distinct peaks in hist data
        x0 = bin_centers[peaks][0] # set start parameters for double gaussian fit 
        a0 = hist[peaks][0]       
        x1 = bin_centers[peaks][1]
        a1 = hist[peaks][1]     
        p0 = [a0, x0, width_start, a1, x1, width_start, offset]

        params, cov = fit_double_gaussian(bin_centers, hist, p0, bounds_double_gaussian)     # replace with curve fit 
        
        
        x_l = min(params[1],params[4])
        x_u = max(params[1],params[4])
        scaling_factor = 1/abs(params[1]-params[4])
        
        trace_norm = (trace-x_l)/(x_u-x_l)
        
        #print(x_l, x_u)
        
        hist_norm, bins_norm = np.histogram(trace_norm, bins = n_bins, density = False)
        bin_centers_norm = 0.5*(bins_norm[1:] + bins_norm[:-1])
        
        p0 = [a0, 0, width_start*scaling_factor, a1, 1, width_start*scaling_factor, offset]
        
        bounds_double_gaussian_norm = ([0,-0.2,0,0,0,0,0],[1e6,0.5,0.5,1e6,1.2,0.5,100])
        
        params_norm, cov_norm = fit_double_gaussian(bin_centers_norm, hist_norm, p0, bounds_double_gaussian_norm)
        snr = snr_calc(params)
        
        return trace_norm, bin_centers_norm, hist_norm, params_norm, snr
        
        
    else: 
        print(len(peaks))
        print(peaks)
        params, snr = 0,0
        
        return 0,0,0,0,0
        
        

def detect_events_vec(x_data, y_data, thresh_upper, thresh_lower): # should be the same function as in code_for_Hubert 
    # make conditions and divide data into points meeting one of those conditions
    above_upper = y_data > thresh_upper
    below_lower = y_data < thresh_lower
    
    # simplify array to values -1,0,1 for the three conditions
    result = np.zeros_like(y_data)
    result[above_upper] = 1
    result[below_lower] = -1
    
    # use diff to detect state changes and set conditions for state changes 
    x_result = x_data[1:]
    diff_result = np.diff(result)    
    diff_events = np.nonzero(diff_result)[0]
    result_events = diff_result[diff_events]
    
    up_idx = np.where((result_events == 2) | ((result_events == 1) & (np.roll(result_events, -1) == 1)))[0]
    up_list = diff_events[up_idx]
    
    down_idx = np.where((result_events == -2) | ((result_events == -1) & (np.roll(result_events, -1) == -1)))[0]
    
    down_list = diff_events[down_idx]
    events_list = np.sort(np.concatenate((up_list, down_list)))
    
    x_events = x_data[events_list]
    times_list = np.diff(x_events)
    
    # check whether first event goes up or down and whether the lists contain events
    if len(up_idx) != 0 and len(down_idx) != 0:
        if up_idx[0] < down_idx[0]:
            up_times = times_list[::2]
            down_times = times_list[1::2]
        elif up_idx[0] > down_idx[0]:
            up_times = times_list[1::2]
            down_times = times_list[::2]   
    else: 
        up_times, down_times = [], []
    
    # Initialize the rectangular signal array
    rectangular_signal = np.zeros_like(y_data)
   
    # Efficiently set the segments of the rectangular signal
    for start, end in zip(up_list, np.append(down_list, len(rectangular_signal))):
        rectangular_signal[start:end] = 1
    for start, end in zip(down_list, np.append(up_list, len(rectangular_signal))):
        rectangular_signal[start:end] = -1
    
    return x_result, diff_result, up_list, down_list, up_times, down_times, rectangular_signal 
    
    
    
def schmitt_errors(a, a_err, params, time, trace):
    
    # errors estimated from variation of a, with a_err being the relative error
    thresh_u = params[4] - a*params[5]
    thresh_l = params[1] + a*params[2]
    
    thresh_u_p = params[4] - a*(1+a_err)*params[5]
    thresh_l_p = params[1] + a*(1+a_err)*params[2]

    thresh_u_m = params[4] - a*(1-a_err)*params[5]
    thresh_l_m = params[1] + a*(1-a_err)*params[2]    
    
    x_result, diff_result, up_list, down_list, up_times, down_times, rectangular_signal = detect_events_vec(time, trace, thresh_u, thresh_l)    
    gamma_up = gamma(up_times)[0]
    gamma_down = gamma(down_times)[0]
        
       
    x_result_p, diff_result_p, up_list_p, down_list_p, up_times_p, down_times_p, rectangular_signal_p = detect_events_vec(time, trace, thresh_u_p, thresh_l_p)    
    gamma_up_p = gamma(up_times_p)[0]
    gamma_down_p = gamma(down_times_p)[0]

    x_result_m, diff_result_m, up_list_m, down_list_m, up_times_m, down_times_m, rectangular_signal_m = detect_events_vec(time, trace, thresh_u_m, thresh_l_m)    
    gamma_up_m = gamma(up_times_m)[0]
    gamma_down_m = gamma(down_times_m)[0]        
    
    gamma_up_s_p = abs(gamma_up_p - gamma_up)
    gamma_up_s_m = abs(gamma_up_m - gamma_up)
    
    gamma_down_s_p = abs(gamma_down_p - gamma_down)
    gamma_down_s_m = abs(gamma_down_m - gamma_down)
    
    # errors estimated based on short events for false positives
    short_times = []
    dt = time[1]-time[0]
    min_down_time = min(down_times)
    min_up_time = min(up_times)
    
    short_up_times = np.where(up_times/dt < 10)
    short_down_times = np.where(down_times == min_down_time)
    
    print("Time distance:")
    print(time[1]-time[0])
    print("Number of short up times:")
    print(min_up_time/dt)
    print(len(short_up_times))
    print("Number of short down times:")
    print(min_down_time/dt)
    print(len(short_down_times))

    
    return gamma_up, gamma_up_p, gamma_up_m, gamma_up_s_p, gamma_up_s_m, gamma_down, gamma_down_p, gamma_down_m, gamma_down_s_p, gamma_down_s_m
    
