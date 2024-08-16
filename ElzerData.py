import h5py
import numpy as np
import itertools
import os
from HDF5Data import HDF5Data


class ElzerData(HDF5Data):

    def __init__(self, t_ini=None, t_load=None, t_read=None, slice_mask=None, n_cycles=None, cycle_length=None,
                 sampling_rate=None, sliced_array=None, read_traces=None, reshaped_data=None, readpath=None,
                 wdir=None, reps=None):
        """
        Initialize the ElzerData object with optional parameters for various attributes.
        Calls the parent class (HDF5Data) initializer.
        """
        super().__init__(readpath=readpath, wdir=wdir)
        self.t_ini = t_ini
        self.t_load = t_load
        self.t_read = t_read
        self.slice_mask = slice_mask
        self.n_cycles = n_cycles
        self.cycle_length = cycle_length
        self.sampling_rate = sampling_rate
        self.sliced_array = sliced_array
        self.reshaped_data = reshaped_data
        self.read_traces = read_traces
        self.reps = reps

    def create_new_file(self, file_name):
        """
        Saves the sliced traces in an HDF5 file.
        """
        try:
            if self.sliced_array is None:
                self.cut_traces()  # Ensure traces are sliced before saving
            file_path = os.path.join(self.wdir, '{}.hdf5'.format(file_name))
            with h5py.File(file_path, 'w') as f:
                # Create group for data
                group = f.create_group('Data')

                # Add TLoad data
                group.create_dataset('TLoad', data=self.t_load)

                # Add ReadTraces data
                for i, trace in enumerate(self.read_traces):
                    time = int(self.t_load[0, i])
                    dataset_name = f'Data/ReadTraces_{time}'
                    ##print(dataset_name)
                    if dataset_name in f:
                        del f[dataset_name]  # Delete the existing dataset
                    group.create_dataset(f'ReadTraces_{time}', data=trace)
        except Exception as e:
            print(f"Error creating file {file_name}: {e}")

    def create_new_file_only_traces(self, file_name):
        """
        Saves the sliced traces in an HDF5 file.
        """
        # try:
        if self.sliced_array is None:
            self.cut_traces()  # Ensure traces are sliced before saving
        file_path = os.path.join(self.wdir, '{}.hdf5'.format(file_name))

        self.read_traces = np.array(self.read_traces)
        self.read_traces = self.read_traces.reshape(-1, self.read_traces.shape[2])



        with h5py.File(file_path, 'w') as f:
            for i, trace in enumerate(self.read_traces):
                #time = int(self.t_load[0, i])
                dataset_name = f'ReadTraces_{i}'
                ##print(dataset_name)
                if dataset_name in f:
                    del f[dataset_name]  # Delete the existing dataset
                f.create_dataset(dataset_name, data=trace)
        
        max_cols = max(array.shape[1] for array in self.sliced_array)

        # Step 2: Initialize an array filled with NaNs
        padded_arrays = np.full((len(self.sliced_array), self.sliced_array[0].shape[0], max_cols), np.nan)

        # Step 3: Copy the data into the new array
        for i, array in enumerate(self.sliced_array):
            padded_arrays[i, :, :array.shape[1]] = array
        
        self.padded_arrays = padded_arrays.reshape(-1, padded_arrays.shape[2])
        single_traces_file_name = f'{file_name}_whole'
        file_path = os.path.join(self.wdir, '{}.hdf5'.format(single_traces_file_name))
        with h5py.File(file_path, 'w') as f:
            for i, trace in enumerate(self.padded_arrays):
                #time = int(self.t_load[0, i])
                dataset_name = f'Traces_{i}'
                ##print(dataset_name)
                if dataset_name in f:
                    del f[dataset_name]  # Delete the existing dataset
                f.create_dataset(dataset_name, data=trace)


        # except Exception as e:
        #     print(f"Error creating file {file_name}: {e}")


    def save_in_file(self):
        """
        Save the sliced traces in an existing HDF5 file.
        """
        try:
            if self.sliced_array is None:
                self.cut_traces()  # Ensure traces are sliced before saving
            group_name = 'SlicedTraces'
            dataset_names = ['TRead']
            datasets = [self.read_traces]
            self.add_group_and_datasets(group_name=group_name, dataset_names=dataset_names, datasets=datasets)
        except Exception as e:
            print(f"Error saving the traces in file {self.readpath}: {e}")

    def set_t_load(self):
        """
        Set the t_load attribute from the file data.
        """
        try:
            if self.file is None:
                self.set_data()  # Load data if not already loaded
            self.set_arrays()
            self.t_load = self.arrays[1]
        except Exception as e:
            print(f"Error setting t_load: {e}")

    def set_n_cycles(self):
        """
        Set the number of pulse cycles per trace.
        """
        try:
            if self.shape_trace is None:
                self.set_trace_shape()
                self.trace_loading_with_reference()
            self.set_t_load()
            self.cycle_length = np.array(self.sampling_rate * (self.t_read + self.t_load[0] + self.t_ini),
                                         dtype=np.int32)
            self.n_cycles = np.array(self.shape_trace[0] / self.cycle_length, dtype=np.int32)
            self.reshape_traces()
        except Exception as e:
            print(f"Error setting n_cycles: {e}")

    def reshape_traces(self):
        """
        Reshape the traces according to cycle length and number of cycles.
        """
        try:
            if self.cycle_length is None:
                self.set_n_cycles()
            if self.trace_reference is None:
                self.trace_loading_with_reference()
            end = self.cycle_length * self.n_cycles
            #print(end)
            shaped_data = [self.trace_reference[:n, 0, self.trace_order[i]].T for i, n in enumerate(end)]
            self.reshaped_data = shaped_data
            # for i, trace in enumerate(self.reshaped_data):
            #     #print(trace)
            #     print(f"Data {i} shape:", trace.shape)
        except Exception as e:
            print(f"Error reshaping traces: {e}")

    def cut_traces(self):
        """
        Cut the traces according to t_load, t_ini, and t_read.
        """
        try:
            self.set_data()
            self.set_trace_shape()
            if self.trace_reference is None:
                self.trace_loading_with_reference()
            if self.n_cycles is None:
                self.set_n_cycles()
            if self.t_load is None:
                self.set_t_load()
            array_length = self.shape_trace[0]
            #print(array_length)
            self.reps = 101
            n_slices = [n * self.reps for n in self.n_cycles]
            #print("n_slices:", n_slices)

            length_array = [np.full(n, self.cycle_length[i]) for i, n in enumerate(n_slices)]
            length_array = np.concatenate(length_array, axis=None)
            #print(length_array)
            #print("length_array shape:", length_array.shape)

            slice_indices = np.cumsum(length_array[:-1])
            #print(slice_indices)
            
            #print("slice_indices shape:", slice_indices.shape)

            flat_traces_con = np.concatenate(self.reshaped_data, axis=None)
            flat_traces = flat_traces_con.flatten()

            #print("flat_traces shape:", flat_traces.shape)

            sliced_traces = np.split(flat_traces, slice_indices)
            self.sliced_traces = sliced_traces

            self.sliced_array = [np.asarray(sliced_traces[i * n:(i + 1) * n]) for i, n in enumerate(n_slices)]
            #self.sliced_array = sliced_traces
            #print("Number of slices:", len(self.sliced_array))
            """
            for i, trace in enumerate(self.sliced_array):
                #print(f"Slice {i} shape:", trace.shape)
            """

            ini_index = np.int32(self.t_ini * self.sampling_rate)
            load_index = np.int32(self.t_load[0] * self.sampling_rate + ini_index)
            #print("ini_index:", ini_index)
            #print("load_index:", load_index)

            read_traces = [read_slice[:, load_index[i]:] for i, read_slice in enumerate(self.sliced_array)]
            # for i, array in enumerate(read_traces):
            #     print(f"Read trace {i} shape:", array.shape)

            self.read_traces = read_traces
        except Exception as e:
            print(f"Error slicing traces: {e}")

    def noise_correction(self):
        """
        Placeholder for noise correction implementation.
        """
        pass

    def detect_events(self):
        """
        Placeholder for event detection implementation.
        """
        pass





