import numpy as np

from sklearn.model_selection import train_test_split

from module.DataTable import DataTable


class DataProcessor():
    
    def __init__(self, validation_fraction=None, test_fraction=None, seed=None, summary=None):
        if summary is None:
            self.validation_fraction = validation_fraction
            self.test_fraction = test_fraction
            self.seed = seed
        else:
            self.validation_fraction = summary.val_split
            self.test_fraction = summary.test_split
            self.seed = summary.seed

    def split_to_train_validate_test(self, data_table, weights=None):
        
        train_idx, test_idx = train_test_split(data_table.df.index,
                                               test_size=self.test_fraction,
                                               random_state=self.seed)
        train = np.asarray([train_idx]).T.flatten()
        test = np.asarray([test_idx]).T.flatten()
        
        train_and_validation_data = DataTable(data_table.df.loc[train])
        test_data = DataTable(data_table.df.loc[test])
        
        test_weights = np.take(weights, test_idx) if weights is not None else None

        if self.validation_fraction > 0:
            train_idx, validation_idx = train_test_split(train_and_validation_data.df.index,
                                                 test_size=self.validation_fraction,
                                                 random_state=self.seed)

            train = np.asarray([train_idx]).T.flatten()
            validation = np.asarray([validation_idx]).T.flatten()

            train_data = DataTable(train_and_validation_data.df.loc[train])
            validation_data = DataTable(train_and_validation_data.df.loc[validation])
            
            train_weights = np.take(weights, train_idx) if weights is not None else None
            validation_weights = np.take(weights, train_idx) if weights is not None else None
        else:
            train_data = DataTable(train_and_validation_data)
            validation_data = None
            train_weights = np.take(weights, train_idx) if weights is not None else None
            validation_weights = None
  
        return train_data, validation_data, test_data, train_weights, validation_weights, test_weights

    def normalize(self, data_table, normalization_type, inverse=False, norm_args=None, scaler=None):
        
        if not isinstance(data_table, DataTable):
            data_table = DataTable(data_table)
        
        if normalization_type == "None":
            return data_table
        
        if normalization_type in ["RobustScaler", "MinMaxScaler", "StandardScaler", "MaxAbsScaler"]:
            if scaler is not None:
                return data_table.normalize(inverse=inverse, scaler=scaler)
            
            data_table.setup_scaler(norm_type=normalization_type, scaler_args=norm_args)
            normalized = data_table.normalize(inverse=inverse)
            normalized.scaler = data_table.scaler
            return normalized
        
        print("ERROR -- Normalization not implemented: ", normalization_type)
        exit(0)
