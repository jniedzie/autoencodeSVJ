from sklearn.decomposition import PCA

import pandas as pd
import numpy as np


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

    def cov_matrix(self, data, verbose=False):
        covariance_matrix = np.cov(data, rowvar=False)
        if self.is_pos_def(covariance_matrix):
            inv_covariance_matrix = np.linalg.inv(covariance_matrix)
            if self.is_pos_def(inv_covariance_matrix):
                return covariance_matrix, inv_covariance_matrix
            else:
                print("Error: Inverse of Covariance Matrix is not positive definite!")
        else:
            print("Error: Covariance Matrix is not positive definite!")

    def MahalanobisDist(self, inv_cov_matrix, mean_distr, data, verbose=False):
        inv_covariance_matrix = inv_cov_matrix
        vars_mean = mean_distr
        diff = data - vars_mean
        md = []
        for i in range(len(diff)):
            md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
        return md

    def MD_detectOutliers(self, dist, extreme=False, verbose=False):
        k = 3. if extreme else 2.
        threshold = np.mean(dist) * k
        outliers = []
        for i in range(len(dist)):
            if dist[i] >= threshold:
                outliers.append(i)  # index of the outlier
        return np.array(outliers)

    def MD_threshold(self, dist, extreme=False, verbose=False):
        k = 3. if extreme else 2.
        threshold = np.mean(dist) * k
        return threshold

    def is_pos_def(self, A):
        if np.allclose(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False

    def __get_model(self):

        pca = PCA(n_components=self.training_params["n_components"],
                  svd_solver=self.training_params["svd_solver"])

        return pca


    def __load_data(self):
        """
        Loading and splitting the data for the training, using data loader and data processor.
        """
        (self.qcd, _, _) = self.data_loader.load_all_data(self.qcd_path, "QCD")
        (self.train_data, self.validation_data, _) = self.data_processor.split_to_train_validate_test(
            data_table=self.qcd)

        ### from tutorial
        # data_dir = '2nd_test'
        # merged_data = pd.DataFrame()
        #
        # for filename in os.listdir(data_dir):
        #     print(filename)
        #     dataset = pd.read_csv(os.path.join(data_dir, filename), sep='\t')
        #     dataset_mean_abs = np.array(dataset.abs().mean())
        #     dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1, 4))
        #     dataset_mean_abs.index = [filename]
        #     merged_data = merged_data.append(dataset_mean_abs)
        #
        # merged_data.columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']

        ###



    
    def __normalize_data(self):
        """
        Preparing normalized version of the training data
        """
        
        print("Trainer scaler: ", self.norm_type)
        print("Trainer scaler args: ", self.norm_args)
        
        self.train_data_normalized = self.data_processor.normalize(data_table=self.train_data,
                                                                   normalization_type=self.norm_type,
                                                                   norm_args=self.norm_args)

        self.validation_data_normalized = self.data_processor.normalize(data_table=self.validation_data,
                                                                        normalization_type=self.norm_type,
                                                                        norm_args=self.norm_args)

        ### this was in the tutorial:
        # self.train_data_normalized.sample(frac=1)

        ### from tutorial
        # scaler = preprocessing.MinMaxScaler()
        #
        # X_train = pd.DataFrame(scaler.fit_transform(dataset_train),
        #                        columns=dataset_train.columns,
        #                        index=dataset_train.index)
        # # Random shuffle training data
        # X_train.sample(frac=1)
        #
        # X_test = pd.DataFrame(scaler.transform(dataset_test),
        #                       columns=dataset_test.columns,
        #                       index=dataset_test.index)


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

        data_train = np.array(X_train_PCA.values)

        _, self.inv_cov_matrix = self.cov_matrix(data_train)
        self.mean_distr = data_train.mean(axis=0)

    
    def get_summary(self):
        """
        @mandatory
        Add additional information to be stored in the summary file. Can return empty dict if
        no additional information is needed.
        """

        flat_inv_cov_matrix = tuple(map(tuple, self.inv_cov_matrix))
        flat_mean_distr = tuple(self.mean_distr)

        summary_dict = {
            'training_output_path': self.training_output_path,
            'qcd_path': self.qcd_path,
            'hlf': True,
            'eflow': True,
            'eflow_base': self.EFP_base,
            'norm_type': self.norm_type,
            'norm_args': self.norm_args,
            'input_dim': self.input_size,
            'inv_cov_matrix': flat_inv_cov_matrix,
            'mean_distr': flat_mean_distr
        }
        
        summary_dict = {**summary_dict, **self.training_params}
        
        return summary_dict
