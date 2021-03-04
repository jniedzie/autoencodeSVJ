import module.utils as utils
from module.Trainer import Trainer
import module.SummaryProcessor as summaryProcessor
from module.DataProcessor import DataProcessor
from module.DataLoader import DataLoader
import numpy as np
import datetime
import pandas as pd
import pickle, os
from pathlib import Path
from module.DataTable import DataTable
from datetime import datetime
from module.PklFile import PklFile
from collections import OrderedDict as odict
import traceback
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint, CSVLogger

from sklearn.ensemble import AdaBoostClassifier

class BdtTrainer:
   
    def __init__(self,
                 qcd_path,
                 signal_path,
                 training_params,
                 output_file_name,
                 EFP_base=None,
                 test_data_fraction=0.2,
                 validation_data_fraction=0.0,
                 norm_type="None",
                 norm_args=None,
                 hlf_to_drop=None,
                 ):
    
        self.seed = np.random.randint(0, 99999999)
        
        utils.set_random_seed(self.seed)

        self.qcd_path = qcd_path
        self.signal_path = signal_path
        self.hlf_to_drop = hlf_to_drop
        self.EFP_base = EFP_base

        self.training_params = training_params
        self.test_data_fraction = test_data_fraction
        self.validation_data_fraction = validation_data_fraction
        self.output_file_name = output_file_name

        self.data_processor = DataProcessor(validation_fraction=self.validation_data_fraction,
                                            test_fraction=self.test_data_fraction,
                                            seed=self.seed)
        
        # Load and split the data
        self.load_data()

        # Normalize the input
        self.norm_type = norm_type
        self.norm_args = norm_args

        print("Trainer scaler: ", self.norm_type)
        print("Trainer scaler args: ", self.norm_args)
        
        self.X_train_normalized = self.data_processor.normalize(data_table=self.X_train,
                                                                normalization_type=self.norm_type,
                                                                norm_args=self.norm_args)

        self.X_test_normalized = self.data_processor.normalize(data_table=self.X_test,
                                                               normalization_type=self.norm_type,
                                                               norm_args=self.norm_args)
        
        # Build the model
        self.model = AdaBoostClassifier(algorithm='SAMME', n_estimators=800, learning_rate=0.5)

    def load_data(self, include_hlf=True, include_eflow=True, hlf_to_drop=['Energy', 'Flavor']):
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
    
        SVJ_Y_train, SVJ_Y_test = [pd.DataFrame(np.ones((len(elt.df), 1)), index=elt.index, columns=['tag']) for elt in [SVJ_X_train, SVJ_X_test]]
        QCD_Y_train, QCD_Y_test = [pd.DataFrame(np.zeros((len(elt.df), 1)), index=elt.index, columns=['tag']) for elt in [QCD_X_train, QCD_X_test]]

        self.X_train = SVJ_X_train.append(QCD_X_train)
        self.Y_train = SVJ_Y_train.append(QCD_Y_train)
    
        self.X_test = SVJ_X_test.append(QCD_X_test)
        self.Y_test = SVJ_Y_test.append(QCD_Y_test)
        
    def run_training(self, training_output_path):
        """
        Runs the training on data loaded and prepared in the constructor, according to training params
        specified in the constructor
        """
        
        self.training_output_path = training_output_path + self.output_file_name
        self.config = PklFile(utils.smartpath(self.training_output_path))
    
        print("\n\nTraining the model")
        print("Filename: ", self.training_output_path)

        self.start_timestamp = datetime.now()
        self.model.fit(self.X_train_normalized, self.Y_train)
        self.end_timestamp = datetime.now()

        defaults = {
            'name': self.training_output_path,
            'trained': True,
            'model_json': '',
            'batch_size': [],
            'epoch_splits': [],
            'metrics': {},
            'time': str(self.end_timestamp - self.start_timestamp),
        }

        for k, v in defaults.items():
            self.config[k] = v
        
        print("model saved")
        
    def save(self, summary_path, model_path):
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
            'norm_args' : self.norm_args,
            'seed': self.seed,
            'start_time': str(self.start_timestamp),
            'end_time': str(self.end_timestamp),
        }
        summaryProcessor.dump_summary_json(self.training_params, summary_dict, output_path=summary_path)

        Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
        pickle.dump(self.model, open(model_path, 'wb'))
