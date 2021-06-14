import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from module.DataProcessor import DataProcessor


class TrainerPca:
    
    def __init__(
            self,
            data_processor,
            data_loader,
            # Architecture specific arguments:
            qcd_path,
            training_params,
            training_output_path,
            EFP_base=None,
            norm_type=None,
            norm_args=None,
            verbose=True
    ):
        """
        Constructor of the specialized Trainer class.
        data_processor and data_loader fields are mandatory and will be passed, ready to be used.
        Names of the remaining arguments match keys of the "training_settings" dict from the config.
        """
        
        # Save data processor and data loader for later use
        self.data_processor = data_processor
        self.data_loader = data_loader
        
        # Save other options passed from the config
        self.qcd_path = qcd_path
        self.training_params = training_params
        self.training_output_path = training_output_path
        self.EFP_base = EFP_base
        self.norm_type = norm_type
        self.norm_args = norm_args
        self.verbose = verbose

        self.inv_cov_matrix = None
        self.mean_distribution = None

        # Load and split the data
        self.__load_data()
        
        # Normalize the input
        self.__normalize_data()
        
        # Build the model
        self.input_size = len(self.qcd.columns)
        self._model = self.__get_model()

    @property
    def model(self):
        """
        @mandatory
        Property that should return the model
        """
        return self._model

    def __is_matrix_positively_defined(self, matrix):
        if np.allclose(matrix, matrix.T):
            try:
                np.linalg.cholesky(matrix)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False

    def __get_inverted_covariance_matrix(self, data):
        covariance_matrix = np.cov(data, rowvar=False)

        if self.__is_matrix_positively_defined(covariance_matrix):
            inv_covariance_matrix = np.linalg.inv(covariance_matrix)
            if self.__is_matrix_positively_defined(inv_covariance_matrix):
                return inv_covariance_matrix
            else:
                print("Error: Inverse of Covariance Matrix is not positive definite!")
        else:
            print("Error: Covariance Matrix is not positive definite!")

    def __get_model(self):

        pca = PCA(n_components=self.training_params["n_components"],
                  svd_solver=self.training_params["svd_solver"])

        return pca

    def __load_data(self):
        """
        Loading and splitting the data for the training, using data loader and data processor.
        """
        self.qcd = self.data_loader.get_data(self.qcd_path, "QCD")
        (self.train_data, self.validation_data, _) = self.data_processor.split_to_train_validate_test(
            data_table=self.qcd)

    def __normalize_data(self):
        """
        Preparing normalized version of the training data
        """
        
        print("Trainer scaler: ", self.norm_type)
        print("Trainer scaler args: ", self.norm_args)
        
        self.train_data_normalized = DataProcessor.normalize(data_table=self.train_data,
                                                                   normalization_type=self.norm_type,
                                                                   norm_args=self.norm_args)

        self.validation_data_normalized = DataProcessor.normalize(data_table=self.validation_data,
                                                                        normalization_type=self.norm_type,
                                                                        norm_args=self.norm_args)

        # Not sure if this is necessary
        self.train_data_normalized.sample(frac=1)

    def train(self):
        """
        @mandatory
        Runs the training of the previously prepared model on the normalized data
        """
        
        print("Filename: ", self.training_output_path)
        print("Number of training samples: ", len(self.train_data_normalized.data))
        print("Number of validation samples: ", len(self.validation_data_normalized.data))
        
        if self.verbose:
            print("\nTraining params:")
            for arg in self.training_params:
                print((arg, ":", self.training_params[arg]))

        X_train_PCA = self.model.fit_transform(self.train_data_normalized)
        X_train_PCA = pd.DataFrame(X_train_PCA)
        X_train_PCA.index = self.train_data_normalized.index

        print("Input size: ", self.input_size)
        print("N components: ", len(X_train_PCA.columns))

        data_train = np.array(X_train_PCA.values)

        self.inv_cov_matrix = self.__get_inverted_covariance_matrix(data_train)
        self.mean_distribution = data_train.mean(axis=0)

    def get_summary(self):
        """
        @mandatory
        Add additional information to be stored in the summary file. Can return empty dict if
        no additional information is needed.
        """

        flat_inv_cov_matrix = tuple(map(tuple, self.inv_cov_matrix))
        flat_mean_distribution = tuple(self.mean_distribution)

        summary_dict = {
            'training_output_path': self.training_output_path,
            'qcd_path': self.qcd_path,
            'include_hlf': True,
            'include_efp': True,
            'efp_base': self.EFP_base,
            'norm_type': self.norm_type,
            'norm_args': self.norm_args,
            'input_dim': self.input_size,
            'inv_cov_matrix': flat_inv_cov_matrix,
            'mean_distribution': flat_mean_distribution
        }
        
        summary_dict = {**summary_dict, **self.training_params}
        
        return summary_dict
