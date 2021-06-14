import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier


class TrainerBdt:
    
    def __init__(
            self,
            data_processor,
            data_loader,
            # Architecture specific arguments:
            qcd_path,
            signal_path,
            training_params,
            training_output_path,
            EFP_base=None,
            norm_type=None,
            norm_args=None,
    ):
        """
        @mandatory
        Constructor of the specialized Trainer class.
        data_processor and data_loader fields are mandatory and will be passed, ready to be used.
        Names of the remaining arguments match keys of the "training_settings" dict from the config.
        """
        
        # Save data processor and data loader for later use
        self.data_processor = data_processor
        self.data_loader = data_loader
        
        # Save other options passed from the config
        self.qcd_path = qcd_path
        self.signal_path = signal_path
        self.training_params = training_params
        self.training_output_path = training_output_path
        self.EFP_base = EFP_base
        self.norm_type = norm_type
        self.norm_args = norm_args
        
        # Load and split the data
        self.__load_data()
        
        # Normalize the input
        self.__normalize_data()
  
        # Build the model
        self.__model = self.__get_model()
    
    @property
    def model(self):
        """
        @mandatory
        Property that should return the model
        """
        return self.__model
    
    def __load_data(self):
        """
        Loading and splitting the data for the training, using data loader and data processor.
        """
        
        QCD = self.data_loader.get_data(self.qcd_path, "QCD")
        SVJ = self.data_loader.get_data(self.signal_path, "SVJ")
        
        (QCD_X_train, _, _) = self.data_processor.split_to_train_validate_test(data_table=QCD)
        (SVJ_X_train, _, _) = self.data_processor.split_to_train_validate_test(data_table=SVJ)
        
        SVJ_Y_train = pd.DataFrame(np.ones((len(SVJ_X_train.df), 1)), index=SVJ_X_train.index, columns=['tag'])
        QCD_Y_train = pd.DataFrame(np.zeros((len(QCD_X_train.df), 1)), index=QCD_X_train.index, columns=['tag'])
        
        self.train_data = SVJ_X_train.append(QCD_X_train)
        self.train_labels = SVJ_Y_train.append(QCD_Y_train)
    
    def __normalize_data(self):
        """
        Preparing normalized version of the training data
        """
        
        print("Trainer scaler: ", self.norm_type)
        print("Trainer scaler args: ", self.norm_args)
    
        self.train_data_normalized = self.data_processor.normalize(data_table=self.train_data,
                                                                   normalization_type=self.norm_type,
                                                                   norm_args=self.norm_args)
    
    def __get_model(self):
        """
        Initializes ML model, using training parameters passed from the config.
        """
        
        model = AdaBoostClassifier(**self.training_params)
        return model
    
    def train(self):
        """
        @mandatory
        Runs the training of the previously prepared model on the normalized data
        """
        print("Filename: ", self.training_output_path)
        self.__model.fit(self.train_data_normalized, self.train_labels)
    
    def get_summary(self):
        """
        @mandatory
        Add additional information to be stored in the summary file. Can return empty dict if
        no additional information is needed.
        """
        summary_dict = {
            'training_output_path': self.training_output_path,
            'qcd_path': self.qcd_path,
            'efp_base': self.EFP_base,
            'norm_type': self.norm_type,
            'norm_args': self.norm_args,
        }
        
        return summary_dict
