import numpy as np
import datetime
import pandas as pd
import pickle, os
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import AdaBoostClassifier


class TrainerBdt:
    
    def __init__(self,
                 qcd_path,
                 signal_path,
                 training_params,
                 training_output_path,
                 data_processor,
                 data_loader,
                 EFP_base=None,
                 norm_type=None,
                 norm_args=None,
                 hlf_to_drop=None,
                 ):
        
        self.qcd_path = qcd_path
        self.signal_path = signal_path
        self.hlf_to_drop = hlf_to_drop
        self.EFP_base = EFP_base
        
        self.training_params = training_params
        self.training_output_path = training_output_path
        
        self.data_processor = data_processor
        self.data_loader = data_loader
        
        # Load and split the data
        self.__load_data()
        
        # Normalize the input
        self.norm_type = norm_type
        self.norm_args = norm_args
        self.__normalize_data()
  
        # Build the model
        self.model = self.__get_model()
    
    def __load_data(self):
        
        (QCD, _, _, _) = self.data_loader.load_all_data(self.qcd_path, "QCD")
        (SVJ, _, _, _) = self.data_loader.load_all_data(self.signal_path, "SVJ")
        
        (QCD_X_train, _, _) = self.data_processor.split_to_train_validate_test(data_table=QCD)
        (SVJ_X_train, _, _) = self.data_processor.split_to_train_validate_test(data_table=SVJ)
        
        SVJ_Y_train = pd.DataFrame(np.ones((len(SVJ_X_train.df), 1)), index=SVJ_X_train.index, columns=['tag'])
        QCD_Y_train = pd.DataFrame(np.zeros((len(QCD_X_train.df), 1)), index=QCD_X_train.index, columns=['tag'])
        
        self.train_data = SVJ_X_train.append(QCD_X_train)
        self.train_labels = SVJ_Y_train.append(QCD_Y_train)
    
    def __normalize_data(self):
        print("Trainer scaler: ", self.norm_type)
        print("Trainer scaler args: ", self.norm_args)
    
        self.train_data_normalized = self.data_processor.normalize(data_table=self.train_data,
                                                                   normalization_type=self.norm_type,
                                                                   norm_args=self.norm_args)
    
    def __get_model(self):
        model = AdaBoostClassifier(**self.training_params)
        return model
    
    def train(self):
        """
        Runs the training on data loaded and prepared in the constructor, according to training params
        specified in the constructor
        """
        
        print("\n\nTraining the model")
        print("Filename: ", self.training_output_path)
        
        self.start_timestamp = datetime.now()
        # self.model.fit(self.train_data_normalized, self.train_labels)
        self.end_timestamp = datetime.now()

        pickle_output_path = self.training_output_path + ".pkl"
        Path(os.path.dirname(pickle_output_path)).mkdir(parents=True, exist_ok=True)
        pickle.dump(self.model, open(pickle_output_path, 'wb'))
        
        print("model saved")
    
    def get_summary(self):
        """
        Add additional information to be stored in the summary file
        """
        summary_dict = {
            'training_output_path': self.training_output_path,
            'qcd_path': self.qcd_path,
            'eflow_base': self.EFP_base,
            'norm_type': self.norm_type,
            'norm_args': self.norm_args,
            'start_time': str(self.start_timestamp),
            'end_time': str(self.end_timestamp),
        }
        
        return summary_dict
