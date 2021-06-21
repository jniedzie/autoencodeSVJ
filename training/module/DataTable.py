from enum import Enum
from glob import fnmatch
import chardet

import sklearn.preprocessing as prep
import pandas as pd
import numpy as np
from ROOT import TFile


class DataTable:
    """
    Wrapper for the pandas data table, allowing data normalization and providing additional columns manipulations.
    It can also store weights for rows in the table.
    """
    
    class NormTypes(Enum):
        MinMaxScaler = 0
        StandardScaler = 1
        RobustScaler = 2
        MaxAbsScaler = 3

    def __init__(self, data, headers=None):
        """ DataTable constructor.
        
        Args:
            data (Any): Data to be put in this data table (pandas Dataframe or something that can be converted to it)
            headers: (Optional) Names of columns, needed if input data is not a Dataframe
        """

        self.df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data, columns=headers)
        self.__update_column_names()

        self.scaler = None
        self.weights = None

    def __getattr__(self, attr):
        return self.df.__getattr__(attr)

    def __getitem__(self, item):
        return self.df[item]

    def setup_scaler(self, norm_type, scaler_args):
        """ Creates scaler and fits it to the data in this data table
        
        Args:
            norm_type (str): Name of the scaler to use for normalization
            scaler_args (Dict[str, Any]): Arguments to be passed to the scaler
        """
        
        norm_type = getattr(self.NormTypes, norm_type)
        self.scaler = getattr(prep, norm_type.name)(**scaler_args)
        self.scaler.fit(self.df)
    
    def normalize(self, inverse=False, scaler=None):
        """ Creates normalized version of this data table
        
        Args:
            inverse (Bool): If true, will apply inverse transformation
            scaler: Scaler object from sklearn.preprocessing (e.g. StandardScaler)

        Returns:
            (DataTable): Normalized data table
        """
        
        if scaler is None:
            if self.scaler is None:
                print("ERROR -- Scaler was not set up before using!!")
                exit(0)
            scaler = self.scaler
        
        data = scaler.inverse_transform(self.df) if inverse else scaler.transform(self.df)
        return DataTable(pd.DataFrame(data, columns=self.df.columns, index=self.df.index))

    def drop_columns(self, columns_to_drop):
        """ Removes specified columns from this data table

        Args:
            columns_to_drop (list[str]): List of columns to drop
        """
    
        to_drop = self.__find_matching_names(columns_to_drop)
        self.df.drop(to_drop, axis=1, inplace=True)

    def remove_empty_rows(self):
        """ Removes rows containing empty jets
        """
        print("Removing empty rows... ", end="")
        self.drop(self[self.Eta == 0].index, inplace=True)
        print("done")

    def calculate_weights(self, weights_path):
        """Calculates jet weights and stores them in self.weights

        Args:
            weights_path (str): Path to the ROOT file with weights
        """
    
        if weights_path is None or weights_path == "":
            self.weights = None
            return
    
        weights_file = TFile.Open(weights_path)
        weights_hist = weights_file.Get("histJetPtWeights")
    
        print("Calculating weights... ", end="")
    
        bin_factor = weights_hist.GetNbinsX() / (weights_hist.GetXaxis().GetXmax() - weights_hist.GetXaxis().GetXmin())
    
        def get_weight(pt):
            return weights_hist.GetBinContent(int(bin_factor * pt) + 1)
    
        weights = self.df[['Pt']].copy()
        weights = weights.apply(lambda x: get_weight(x.Pt), 1)
    
        print("done")
    
        self.weights = np.array(weights)
    
    def __find_matching_names(self, input_list):
        """ Finds elements of provided list matching names of columns in this data table.
        
        Args:
            input_list (list[str]): Input list to filter

        Returns:
            list[str]: List of names matching column names in this data table
        """
        
        match_list = list(self.df.columns)
        match = set()
    
        for g in input_list:
            match.update(fnmatch.filter(match_list, g))
    
        return list(match)
    
    def __update_column_names(self):
        """ Adds 'efp' prefix to EFP columns and decodes byte strings for other columns.
        """
        
        new_names = dict()
    
        for column in self.df.columns:
            new_name = column if type(column) is str else column.decode(chardet.detect(column)["encoding"])
            new_names[column] = "efp %s" % new_name if column.isdigit() else new_name
    
        self.df.rename(columns=new_names, inplace=True)
