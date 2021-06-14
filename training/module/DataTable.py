from enum import Enum
from glob import fnmatch
import chardet

import sklearn.preprocessing as prep
import pandas as pd
import numpy as np


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

    table_count = 0
    
    def __init__(self, data, headers=None):
        """ DataTable constructor.
        
        Args:
            data: Data to be put in this data table (supports multiple types)
            headers: (Optional) Names of columns
        """
        
        self.name = "Table {}".format(DataTable.table_count)
        self.scaler = None
        DataTable.table_count += 1
        
        # TODO: this can clearly be further simplified...
        if headers is not None:
            self.headers = headers
            data = np.asarray(data)
            if len(data.shape) < 2:
                data = np.expand_dims(data, 1)
            self.data = data
        elif isinstance(data, pd.DataFrame):
            self.headers = data.columns
            self.data = data
        elif isinstance(data, DataTable):
            self.headers = data.headers
            self.data = data.df.values
            self.name = data.name
        
        assert len(self.data.shape) == 2, "data must be matrix!"
        assert len(self.headers) == self.data.shape[1], "n columns must be equal to n column headers"
        assert len(self.data) > 0, "n samples must be greater than zero"
        
        if isinstance(self.data, pd.DataFrame):
            self.df = self.data
            self.data = self.df.values
        else:
            self.df = pd.DataFrame(self.data, columns=self.headers)
            
        self.__update_column_names()
        self.weights = None
    
    def __getattr__(self, attr):
        if hasattr(self.df, attr):
            return self.df.__getattr__(attr)
        else:
            raise AttributeError

    def __getitem__(self, item):
        return self.df[item]

    def __str__(self):
        return self.df.__str__()

    def __repr__(self):
        return self.df.__repr__()
    
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
        self.headers = list(self.df.columns)
        self.data = np.asarray(self.df)

    def merge_columns(self, other):
        """ Appends columns of other data table to this one

        Args:
            other (DataTable): Data table to append to this one

        Returns:
            (DataTable): Merged data table
        """
    
        assert self.shape[0] == other.shape[0], 'data tables must have same number of samples'
        return DataTable(self.df.join(other.df))
        
    def __find_matching_names(self, input_list):
        """ Finds elements of provided list matching names of columns in this data table.
        
        Args:
            input_list (list[str]): Input list to filter

        Returns:
            (list[str]): List of names matching column names in this data table
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
        self.headers = list(self.df.columns)
