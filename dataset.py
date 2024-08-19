import h5py
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class SimDataset(Dataset):
    """
    A custom dataset class for loading simulation data stored in an HDF5 file.
    """
    def __init__(self, file_path, scale_transform=None, noise_transform=None):
        """
        Initialize the dataset with the path to the HDF5 file and an optional transform.

        Parameters:
        file_path (str): Path to the HDF5 file.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(Dataset, self).__init__()  # Initialize the parent Dataset class
        self.file_path = file_path  # Store the file path
        self.scale_transform = scale_transform  # Store the transform
        self.noise_transform = noise_transform
        
        file = h5py.File(file_path, 'r')  # Open the HDF5 file in read mode
        self.keys = list(file.keys())  # Get the list of keys (datasets) in the file
        random.shuffle(self.keys)  # Shuffle the keys to randomize the order of access
        self.traces = {key: np.array(file[key], dtype=np.float32) for key in self.keys}

        file.close()  # Close the HDF5 file

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
        int: Number of samples.
        """
        return len(self.keys)

    def __getitem__(self, index):
        """
        Retrieve a sample from the dataset at the specified index.

        Parameters:
        index (int): Index of the sample to retrieve.

        Returns:
        torch.Tensor: The sample as a tensor.
        """
        # file = h5py.File(self.file_path, 'r')  # Open the HDF5 file in read mode
        # key = self.keys[index]  # Get the key corresponding to the given index
        # clean_trace = np.array(file[key]) # Load the data for the given key and convert it to a numpy array
        # file.close()  # Close the HDF5 file
        clean_trace = self.traces[self.keys[index]]

        if self.noise_transform:
            noisy_trace = self.noise_transform(clean_trace)
        else:
            noisy_trace = clean_trace.copy()
            #print('No noise transform! Data will be labeled to itself!')

        if self.scale_transform:  # Apply the scale transform if it exists
            #clean_trace = self.scale_transform(clean_trace)
            noisy_trace = self.scale_transform(noisy_trace)
        else:
            clean_trace = clean_trace.reshape(1, -1)
            noisy_trace = noisy_trace.reshape(1, -1)
            #print(np.shape(noisy_trace))
        return torch.tensor(noisy_trace, dtype=torch.float32), torch.tensor(clean_trace, dtype=torch.float32) # Convert the numpy array to a PyTorch tensor and return it

class MeasuredNoise(object):
    def __init__(self, noise_traces, amps, amps_dist=None):
        self.amps = amps 
        self.amps_dist = amps_dist
        self.noise_traces = noise_traces

    def __call__(self, trace, noise_scaling=None):
        
        amp = np.random.choice(self.amps, p=self.amps_dist)
        #print(amp)
        i_noise = np.random.randint(self.noise_traces.shape[0])
        j_noise = np.random.randint(self.noise_traces.shape[1])
        noise_trace = self.noise_traces[i_noise, j_noise]

        start = np.random.randint(0, len(noise_trace)-len(trace))
        stop = start + len(trace)


        noise = noise_trace[start:stop]
        noise -= np.mean(noise)
        noise /= np.max(np.abs(noise))
        noisy_trace = trace + amp*noise
        return noisy_trace



class Noise(object):
    def __init__(self, shape, sim_t, sigma, amps, freqs, weights_sigma=None, weights_amps=None):
        """
        Initialize the Noise object with the specified parameters.

        Parameters:
        shape (tuple): Shape of the noise to be generated.
        sigma (float): Standard deviation for noise generation.
        sim_t (float): Total simulation time.
        """
        self.sigma_pos = sigma
        self.shape = shape
        self.T = sim_t
        self.amps_pos = amps
        self.freqs = freqs
        
        self.weights_sigma = weights_sigma
        self.weights_amps = weights_amps

    def __call__(self, trace, noise_scaling=None):
        """
        Generate noisy trace by adding white noise, pink noise, and interference signal to the input trace.

        Parameters:
        trace (array-like): The input trace to which noise will be added.
        noise_scaling (float, optional): Scaling factor for pink noise. Default is None.

        Returns:
        array-like: The noisy trace.
        """
        if hasattr(self.sigma_pos, "__len__"):
            if hasattr(self.weights_sigma, "__len__"): 
                
                self.sigma = np.random.choice(self.sigma_pos, p=self.weights_sigma)
            else:            
                self.sigma = np.random.choice(self.sigma_pos)
        else: 
            self.sigma = self.sigma_pos

        self.white_sigma = self.sigma
        self.pink_sigma = 0.1 * self.sigma

        if len(self.amps_pos.shape) == 2:
            if len(self.weights_amps.shape) == 2:
                self.amps = [np.random.choice(self.amps_pos[i, :], p=self.weights_amps[i, :]) for i in range(self.amps_pos.shape[0])] 
            else:
                self.amps = [np.random.choice(self.amps_pos[i, :]) for i in range(self.amps_pos.shape[0])] 

        if noise_scaling is not None:
            self.pink_sigma = noise_scaling * self.white_sigma

        # Generate different types of noise
        white_noise = self.generate_white_noise()
        pink_noise = self.generate_pink_noise()
        interference_signal = self.interference_signal()

        # Add the generated noises to the input trace
        noisy_trace = trace + white_noise + pink_noise + interference_signal
        return noisy_trace

    def generate_white_noise(self):
        """
        Generate white noise with a given standard deviation and number of samples.

        Returns:
        array-like: The generated white noise.
        """
        white_noise = np.random.normal(0.0, self.white_sigma, self.shape)
        return white_noise

    def generate_pink_noise(self):
        """
        Generate pink noise using the inverse FFT method.

        Returns:
        array-like: The generated pink noise.
        """
        exponents = np.fft.fftfreq(self.shape)
        exponents[0] = 1  # Avoid division by zero
        amplitudes = 1 / np.sqrt(np.abs(exponents))
        amplitudes[0] = 0  # Set the DC component to 0
        random_phases = np.exp(2j * np.pi * np.random.random(self.shape))
        pink_noise_spectrum = amplitudes * random_phases
        pink_noise = np.fft.ifft(pink_noise_spectrum).real
        return pink_noise

    def interference_signal(self):
        """
        Generate an interference signal consisting of sinusoidal components with noise.

        Parameters:
        amps (array-like): Amplitudes of the sinusoidal components.
        freqs (array-like): Frequencies of the sinusoidal components.

        Returns:
        array-like: The generated interference signal.
        """
        interference_noise = 0
        t = np.linspace(0, self.T, self.shape)
        phi_0 = np.random.uniform(0, 2 * np.pi, 1)  # Random initial phase

        for i, amp in enumerate(self.amps):
            interference_noise += (amp * np.sin(2 * np.pi * self.freqs[i] * t + phi_0) +
                                   self.generate_white_noise() +
                                   self.pink_sigma * self.generate_pink_noise())

        return interference_noise


class MinMaxScalerTransform:
    def __init__(self, feature_range=(0, 1)):
        """
        Initialize the MinMaxScalerTransform with a given feature range.

        Parameters:
        feature_range (tuple): Desired range of transformed data.
        """
        #self.scaler = MinMaxScaler(feature_range=feature_range)
        self.scaler = StandardScaler()
        
    def fit_data(self, data):
        """
        Fit the MinMaxScaler to the data.

        Parameters:
        data (numpy array): The data to fit the scaler on.

        Returns:
        self: Returns an instance of self.
        """
        self.scaler.fit(data)
        return self

    def fit_from_hdf5(self, file_path):
        """
        Fit the MinMaxScaler to the data from an HDF5 file.

        Parameters:
        file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset in the HDF5 file.

        Returns:
        self: Returns an instance of self.
        """
        with h5py.File(file_path, 'r') as file:
            all_keys = file.keys()
            data = np.array([file[key] for key in all_keys],dtype=np.float32)
        self.scaler.fit(data)
        return self

    def transform_data(self, data):
        """
        Transform the data using the fitted MinMaxScaler.

        Parameters:
        data (numpy array): The data to transform.

        Returns:
        scaled_data (numpy array): The transformed data as a PyTorch tensor.
        """
        data = data.reshape(1, -1)
        scaled_data = self.scaler.transform(data)
        return scaled_data

    def fit_transform(self, data):
        """
        Fit the MinMaxScaler to the data, then transform it.

        Parameters:
        data (numpy array): The data to fit and transform.

        Returns:
        rescaled data (numpy array): The transformed data as a PyTorch tensor.
        """
        self.fit_data(data)
        return self.transform_data(data)

    def __call__(self, data):
        """
        Apply the transform to a PyTorch tensor.

        Parameters:
        tensor (torch.Tensor): The data to transform.

        Returns:
        scaled data (numpy array): The transformed data as a PyTorch tensor.
        """
        #data = data.reshape(1, -1)
        scaled_data = self.transform_data(data)
        return scaled_data#.reshape(-1, )


"""
class Rescale(object):

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, trace):
        pass
"""

