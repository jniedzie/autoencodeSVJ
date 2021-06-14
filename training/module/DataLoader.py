from collections import OrderedDict
from glob import glob

import h5py
import numpy as np
from ROOT import TFile

from module.DataTable import DataTable


class DataLoader:
    """
    data loader/handler/merger for h5 files with the general format of this repository
    """
    
    def __init__(self, name=""):
        self.name = name
        self.samples = OrderedDict()
        self.sample_keys = None
        self.data = OrderedDict()
        self.labels = OrderedDict()
        self.weights = {}

        self.include_hlf = None
        self.include_epf = None
        self.include_constituents = None
        self.hlf_to_drop = None
        self.efp_to_drop = None
        self.constituents_to_drop = None
        self.max_jets = None

    def set_params(self,
                   include_hlf, include_efp, include_constituents,
                   hlf_to_drop, efp_to_drop, constituents_to_drop, max_jets):
        self.include_hlf = include_hlf
        self.include_epf = include_efp
        self.include_constituents = include_constituents
        self.hlf_to_drop = hlf_to_drop
        self.efp_to_drop = efp_to_drop
        self.constituents_to_drop = constituents_to_drop
        self.max_jets = int(max_jets)
        
    def load_all_data(self, data_path, name, weights_path=None):
        """
        Args:
            data_path (str): Path to data to load (can contain wildcards)
            name (str): desc
            weights_path (str): desc

        Returns:
            (tuple): tuple containing:
                servers(list) servers to use
                msg (str): logging message string
        """
    
        to_include = []
        if self.include_hlf:
            to_include.append("jet_features")
    
        if self.include_epf:
            to_include.append("jet_eflow_variables")

        if self.include_constituents:
            to_include.append("jet_constituents")

        if not (self.include_hlf or self.include_epf):
            raise AttributeError
    
        files = glob(data_path)

        if len(files) == 0:
            print("\n\nERROR -- no files found in ", data_path, "\n\n")
            raise AttributeError
        
        for f in files:
            self.__add_sample(f)
    
        event = self.__make_table('event_features', name + ' event features')

        data = self.__make_tables(to_include, name)
        data.drop(data[data.Eta == 0].index, inplace=True)
        
        self.__calculate_weights(data, weights_path, name)
        
        columns_to_drop = ['efp 0'] + self.hlf_to_drop + self.efp_to_drop + self.constituents_to_drop
        data.drop_columns(columns_to_drop)

        return data, event

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
    
    def __add_sample(self, sample_path):
        """
        
        Args:
            sample_path (str):

        Returns:
            (void)
        """
        
        if sample_path in self.samples:
            return
            
        with h5py.File(sample_path, mode="r") as f:
            
            print("Adding sample ", sample_path)
            self.samples[sample_path] = f
            
            keys = set(f.keys())
            
            if self.sample_keys is None:
                self.sample_keys = keys
            elif keys != self.sample_keys:
                print("ERROR -- different h5 samples seem to have different groups/keys")
                exit()
            
            self.__update_data(f)
    
    def __make_table(self, key, name=None):
        """ stack, combine, or split """
        assert key in self.sample_keys
        
        data = self.data[key]
        labels = self.labels[key]
        name = name or self.name
        
        if len(data.shape) == 1:
            return DataTable(np.expand_dims(data, 1), headers=labels, name=name)
        elif len(data.shape) == 2:
            return DataTable(data, headers=labels, name=name)
        elif len(data.shape) == 3:
            return DataTable(np.vstack(data), headers=labels, name=name)
        elif len(data.shape) == 4:

            constituents_labels = []
            for i_constituent in range(len(data[0][0])):
                for label in labels:
                    constituents_labels.append(np.bytes_("constituent_")+label+np.bytes_("_"+str(i_constituent)))

            stacked_data = np.vstack(data)
            stacked_data = stacked_data.transpose((1, 2, 0))
            stacked_data = np.vstack(stacked_data)
            stacked_data = stacked_data.transpose()

            return DataTable(stacked_data, headers=constituents_labels, name=name)

        else:
            raise AttributeError
    
    def __make_tables(self, key_list, name):
        """Prepares a table containing all information matching provided key_list
        
        Args:
            key_list (list): List of h5 file keys to include in the output table
            name: Name of the output table

        Returns:
            (DataTable): Table containing all information matching provided key_list
        """
        
        tables = [self.__make_table(k) for k in key_list]

        ret, tables = tables[0], tables[1:]
        for table in tables:
            ret = ret.merge_columns(table, name)
        return ret
       
    def __update_data(self, sample_file):
        
        for key in sample_file.keys():
            if 'data' not in sample_file[key] or 'labels' not in sample_file[key]:
                print("ERROR -- group ", key, " doesn't contain 'data' or 'labels'")
                exit()
            
            if key not in self.labels:
                self.labels[key] = np.asarray(sample_file[key]['labels'])
            else:
                assert (self.labels[key] == np.asarray(sample_file[key]['labels'])).all()

            sample_data = sample_file[key]['data']

            if key == "jet_features" or key == "jet_eflow_variables":
                sample_data = sample_data[:, 0:self.max_jets, :]

            if key == "jet_constituents":
                sample_data = sample_data[:, 0:self.max_jets, :, :]

            if key not in self.data:
                self.data[key] = np.asarray(sample_data)
            else:
                self.data[key] = np.concatenate([self.data[key], sample_data])
