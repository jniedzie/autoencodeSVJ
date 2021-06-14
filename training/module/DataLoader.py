from collections import OrderedDict
from glob import glob

import h5py
import numpy as np
from ROOT import TFile

from module.DataTable import DataTable


class DataLoader:
    """
    Allows to load data from h5 files as data table, dropping unused variables and limiting number of jets per event.
    """
    
    def __init__(self, variables_to_drop, max_jets):
        """ DataLoader constructor.
        
        Args:
            variables_to_drop (List[str]): List of variables to drop.
            max_jets (int): Maximum number of jets per event to load.
        """
        
        self.variables_to_drop = variables_to_drop
        self.max_jets = max_jets
        
        self.sample_keys = None
        self.data = OrderedDict()
        self.labels = OrderedDict()
        self.weights = {}
        self.already_added_paths = []
        
    def get_data(self, data_path, name, weights_path=None, per_event=False):
        """ Loads data from provided path and returns as a data table
        Args:
            data_path (str): Path to data to load (can contain wildcards)
            name (str): Output data table name
            weights_path (str): If specified, will load weights histogram and calculate weights that can be later
                                accessed via self.weights
            per_event (Bool): If true, data table will be organized per-event rather than per-jet

        Returns:
            (DataTable)
        """

        keys_to_skip = [] if per_event else ["event_features"]
    
        for f in DataLoader.__get_files_from_path(data_path):
            self.__add_sample(f, keys_to_skip)
    
        if per_event:
            data = self.__make_table("event_features")
        else:
            data = self.__make_tables()
            data.drop(data[data.Eta == 0].index, inplace=True)  # removes empty jets
            self.__calculate_weights(data, weights_path, name)
            data.drop_columns(self.variables_to_drop)
    
        return data

    def __calculate_weights(self, data, weights_path, name):
        """ Calculates jet weights and stores them in self.weights
        
        Args:
            data (DataTable): Table of jets for which weights will be calculated
            weights_path (str): Path to the ROOT file with weights
            name (str): Name of the sample under which jet weights will be stored
        """
        
        if weights_path is None or weights_path == "":
            self.weights[name] = None
            return
    
        weights_file = TFile.Open(weights_path)
        weights_hist = weights_file.Get("histJetPtWeights")
    
        print("Calculating weights...", end="")
        weights = [weights_hist.GetBinContent(weights_hist.GetXaxis().FindFixBin(entry.Pt)) for _, entry in
                   data.df.iterrows()]
        print("done")
        self.weights[name] = np.array(weights)
    
    def __add_sample(self, sample_path, keys_to_skip):
        """
        Adds data from h5 file to self.samples dict. Stores h5 keys in self.sample_keys, or checks that
        they are consistent with the existing ones.
        
        Args:
            sample_path (str): Path to h5 file to be added.
        """
        
        if sample_path in self.already_added_paths:
            return
            
        with h5py.File(sample_path, mode="r") as h5_file:
            
            print("Adding sample ", sample_path)
            self.already_added_paths.append(sample_path)
            
            keys = set(h5_file.keys())
            
            if self.sample_keys is None:
                self.sample_keys = keys
            elif keys != self.sample_keys:
                print("ERROR -- different h5 samples seem to have different groups/keys")
                exit()
            
            self.__add_data_and_labels(h5_file, keys_to_skip)
    
    def __make_table(self, key):
        """
        Creates a data table for given key, reshaping depending on whether these are event-level, jet-level
        or constituent-level variables.
        
        Args:
            key (str): Key to process (e.g. jet_features)

        Returns:
            (DataTable): Properly shaped data table
        """
        assert key in self.sample_keys
        
        data = self.data[key]
        labels = self.labels[key]
        
        if len(data.shape) == 1:
            # some per-sample variables (?)
            return DataTable(np.expand_dims(data, 1), headers=labels)
        elif len(data.shape) == 2:
            # events features (?)
            return DataTable(data, headers=labels)
        elif len(data.shape) == 3:
            # jet features
            return DataTable(np.vstack(data), headers=labels)
        elif len(data.shape) == 4:
            # jet constituents

            constituents_labels = []
            for i_constituent in range(len(data[0][0])):
                for label in labels:
                    constituents_labels.append(np.bytes_("constituent_")+label+np.bytes_("_"+str(i_constituent)))

            stacked_data = np.vstack(data)
            stacked_data = stacked_data.transpose((1, 2, 0))
            stacked_data = np.vstack(stacked_data)
            stacked_data = stacked_data.transpose()

            return DataTable(stacked_data, headers=constituents_labels)
        else:
            raise AttributeError
    
    def __make_tables(self):
        """Prepares a table containing all jet-level information
        
        Returns:
            (DataTable): Table containing all jet-level information
        """
        
        tables = [self.__make_table(k) for k in self.sample_keys if k != "event_features"]

        ret, tables = tables[0], tables[1:]
        for table in tables:
            ret = ret.merge_columns(table)
        return ret
       
    def __add_data_and_labels(self, h5_file, keys_to_skip):
        """ Adds data and labels (variable names) from h5 file
        
        Args:
            h5_file: h5 file to be added
        """
        keys = [k for k in h5_file.keys() if k not in keys_to_skip]
        
        for key in keys:
            DataLoader.__check_file_ok(h5_file, key)
            
            if key not in self.labels:
                self.labels[key] = np.asarray(h5_file[key]['labels'])

            sample_data = h5_file[key]['data']
            sample_data = self.__h5_to_array(sample_data, key)
            
            if key not in self.data:
                self.data[key] = np.asarray(sample_data)
            else:
                self.data[key] = np.concatenate([self.data[key], sample_data])

    def __h5_to_array(self, data, key):
        """ Converts h5 dataset to array, limiting number of jets per event to self.max_jets
        
        Args:
            data: Input h5 dataset
            key (str): h5 group this dataset belongs to

        Returns:
            (np.ndarray)
        """
        
        if key in ["jet_features", "jet_eflow_variables"]:
            return data[:, 0:self.max_jets, :]
        if key == "jet_constituents":
            return data[:, 0:self.max_jets, :, :]

        print("ERROR -- no known way to reshape group ", key)
        exit()

    @staticmethod
    def __check_file_ok(h5_file, key):
        """ Verifies that h5 file looks healthy for given key. If not, quits application.
        Args:
            h5_file: h5 file to test
            key (str): h5 key to check
        """
        
        if 'data' not in h5_file[key] or 'labels' not in h5_file[key]:
            print("ERROR -- group ", key, " doesn't contain 'data' or 'labels'")
            exit()

    @staticmethod
    def __get_files_from_path(path):
        """
        Returns global paths to files from provided path (can contain wildcards).
        If no files were found, quits application.
        
        Args:
            path (str): Path to files

        Returns:
            (List[str]): List of global paths to files
        """
        
        files = glob(path)
    
        if len(files) == 0:
            print("ERROR -- no files found in ", path)
            exit()
            
        return files
