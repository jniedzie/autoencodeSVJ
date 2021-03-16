import module.utils as utils
from module.DataLoader import DataLoader

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
                 output_file_name,
                 training_output_path,
                 data_processor,
                 seed,
                 test_data_fraction,
                 validation_data_fraction,
                 EFP_base=None,
                 norm_type=None,
                 norm_args=None,
                 hlf_to_drop=None,
                 ):
        self.seed = seed
        utils.set_random_seed(self.seed)
        
        self.qcd_path = qcd_path
        self.signal_path = signal_path
        self.hlf_to_drop = hlf_to_drop
        self.EFP_base = EFP_base
        
        self.training_params = training_params
        self.test_data_fraction = test_data_fraction
        self.validation_data_fraction = validation_data_fraction
        self.output_file_name = output_file_name
        self.training_output_path = training_output_path
        
        self.data_processor = data_processor
        
        # Load and split the data
        self.__load_data()
        
        # Normalize the input
        self.norm_type = norm_type
        self.norm_args = norm_args
        
        print("Trainer scaler: ", self.norm_type)
        print("Trainer scaler args: ", self.norm_args)
        
        self.X_train_normalized = data_processor.normalize(data_table=self.X_train,
                                                                normalization_type=self.norm_type,
                                                                norm_args=self.norm_args)
        
        self.X_test_normalized = data_processor.normalize(data_table=self.X_test,
                                                               normalization_type=self.norm_type,
                                                               norm_args=self.norm_args)
        
        # Build the model
        self.model = AdaBoostClassifier(algorithm='SAMME', n_estimators=800, learning_rate=0.5)
    
    def __load_data(self, include_hlf=True, include_eflow=True, hlf_to_drop=['Energy', 'Flavor']):
        data_loader = DataLoader()
        
        (QCD, _, _, _) = data_loader.load_all_data(self.qcd_path, "QCD",
                                                   include_hlf=include_hlf,
                                                   include_eflow=include_eflow,
                                                   hlf_to_drop=hlf_to_drop)
        
        (SVJ, _, _, _) = data_loader.load_all_data(self.signal_path, "SVJ",
                                                   include_hlf=include_hlf,
                                                   include_eflow=include_eflow,
                                                   hlf_to_drop=hlf_to_drop)
        
        (QCD_X_train, _, QCD_X_test, _, _) = self.data_processor.split_to_train_validate_test(data_table=QCD)
        (SVJ_X_train, _, SVJ_X_test, _, _) = self.data_processor.split_to_train_validate_test(data_table=SVJ)
        
        SVJ_Y_train, SVJ_Y_test = [pd.DataFrame(np.ones((len(elt.df), 1)), index=elt.index, columns=['tag']) for elt in
                                   [SVJ_X_train, SVJ_X_test]]
        QCD_Y_train, QCD_Y_test = [pd.DataFrame(np.zeros((len(elt.df), 1)), index=elt.index, columns=['tag']) for elt in
                                   [QCD_X_train, QCD_X_test]]
        
        self.X_train = SVJ_X_train.append(QCD_X_train)
        self.Y_train = SVJ_Y_train.append(QCD_Y_train)
        
        self.X_test = SVJ_X_test.append(QCD_X_test)
        self.Y_test = SVJ_Y_test.append(QCD_Y_test)
    
    def train(self):
        """
        Runs the training on data loaded and prepared in the constructor, according to training params
        specified in the constructor
        """
        
        self.training_output_path = self.training_output_path + self.output_file_name
        
        print("\n\nTraining the model")
        print("Filename: ", self.training_output_path)
        
        self.start_timestamp = datetime.now()
        self.model.fit(self.X_train_normalized, self.Y_train)
        self.end_timestamp = datetime.now()

        pickle_output_path = self.training_output_path + ".pkl"
        Path(os.path.dirname(pickle_output_path)).mkdir(parents=True, exist_ok=True)
        pickle.dump(self.model, open(pickle_output_path, 'wb'))
        
        print("model saved")
    
    def get_summary(self):
        """
        Dumps summary of the most recent training to a summary file.
        """
        
        summary_dict = {
            'training_output_path': self.training_output_path,
            'qcd_path': self.qcd_path,
            'hlf': True,
            'hlf_to_drop': tuple(self.hlf_to_drop),
            'eflow': True,
            'eflow_base': self.EFP_base,
            'test_split': self.test_data_fraction,
            'val_split': self.validation_data_fraction,
            'norm_type': self.norm_type,
            'norm_args': self.norm_args,
            'seed': self.seed,
            'start_time': str(self.start_timestamp),
            'end_time': str(self.end_timestamp),
        }
        
        return summary_dict