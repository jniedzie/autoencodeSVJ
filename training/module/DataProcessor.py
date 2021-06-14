import numpy as np

from sklearn.model_selection import train_test_split

from module.DataTable import DataTable


class DataProcessor:
    """
    Allows to split data into training, validation and test subsets, as well as normalize the data.
    """
    
    def __init__(self, validation_fraction=None, test_fraction=None, seed=None, summary=None):
        """
        Constructor of DataProcessor. Sets validation and test fractions (either directly or from summary), as
        well as the random seed.
        
        Args:
            validation_fraction (float): Fraction of data used for validation
            test_fraction (float): Fraction of data used for testing
            seed (int): Random seed
            summary: If specified, summary will be used instead of other arguments
        """
        
        if summary is None:
            self.validation_fraction = validation_fraction
            self.test_fraction = test_fraction
            self.seed = seed
        else:
            self.validation_fraction = summary.val_split
            self.test_fraction = summary.test_split
            self.seed = summary.seed

    def split_to_train_validate_test(self, input_data):
        """
        Splits input data into train, validation and test subsets, according to fractions set in the constructor or
        loaded from summary. Also makes sure that corresponding weights are stored in the data table.
        
        Args:
            input_data (DataTable): Input data table to be split.

        Returns:
            (tuple[DataTable, DataTable, DataTable]): Tuple containing train, validation and test parts of the
                provided data table, respectively.
        """
        
        train_idx, test_idx = train_test_split(input_data.df.index,
                                               test_size=self.test_fraction,
                                               random_state=self.seed)
        
        train_and_validation_data = DataProcessor.__get_datatable_from_indices(input_data, train_idx)
        
        test_data = DataProcessor.__get_datatable_from_indices(input_data, test_idx)

        if self.validation_fraction > 0:
            train_idx, validation_idx = train_test_split(train_and_validation_data.df.index,
                                                         test_size=self.validation_fraction,
                                                         random_state=self.seed)

            train_data = DataProcessor.__get_datatable_from_indices(train_and_validation_data, train_idx)
            validation_data = DataProcessor.__get_datatable_from_indices(train_and_validation_data, validation_idx)
        else:
            train_data = DataTable(train_and_validation_data)
            validation_data = None
            
        return train_data, validation_data, test_data

    @staticmethod
    def normalize(data, normalization_type, inverse=False, norm_args=None, scaler=None):
        """
        Applies (inverse) normalization to the data.
        
        Args:
            data (DataTable): Data to be normalized
            normalization_type (str): Name of the scaler class (or "None")
            inverse (bool): If True, inverse transformation will be applied
            norm_args (dict[str, Any]): Arguments to be passed to the scaler
            scaler: If provided, this scaler object will be used for normalization instead of creating a new one.
        """
        
        if not isinstance(data, DataTable):
            data = DataTable(data)
        
        if normalization_type == "None":
            return data
        
        if normalization_type in ["RobustScaler", "MinMaxScaler", "StandardScaler", "MaxAbsScaler"]:
            if scaler is not None:
                return data.normalize(inverse=inverse, scaler=scaler)
            
            data.setup_scaler(norm_type=normalization_type, scaler_args=norm_args)
            normalized = data.normalize(inverse=inverse)
            normalized.scaler = data.scaler
            return normalized
        
        print("ERROR -- Normalization not implemented: ", normalization_type)
        exit(0)

    @staticmethod
    def __get_datatable_from_indices(input_data, indices):
        """
        Selects rows in data table corresponding to provided indices. Takes care of setting weights as well.
        
        Args:
            input_data (DataTable): Data table to pick entries from
            indices (Int64Index): Indices to keep in the output data table.

        Returns:
            (DataTable)
        """
        data = DataTable(input_data.df.loc[np.asarray([indices]).T.flatten()])
        if input_data.weights is not None:
            data.weights = np.take(input_data.weights, indices)
        
        return data
