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

        # SVJ_Y_train = DataTable(SVJ_Y_train)
        # SVJ_Y_test = DataTable(SVJ_Y_test)
        # QCD_Y_train = DataTable(QCD_Y_train)
        # QCD_Y_test = DataTable(QCD_Y_test)
    
        self.X_train = SVJ_X_train.append(QCD_X_train)
        self.Y_train = SVJ_Y_train.append(QCD_Y_train)
    
        self.X_test = SVJ_X_test.append(QCD_X_test)
        self.Y_test = SVJ_Y_test.append(QCD_Y_test)
        
        
    def run_training(self, training_output_path, summaries_path, verbose=False):
        """
        Runs the training on data loaded and prepared in the constructor, according to training params
        specified in the constructor
        """
        
        self.training_output_path = training_output_path + self.output_file_name
    
        print("\n\nTraining the model")
        print("Filename: ", self.training_output_path)
        
        if verbose:
            if hasattr(self.model, "summary"):
                self.model.summary()
            # print("\nTraining params:")
            # for arg in self.training_params:
            #     print((arg, ":", self.training_params[arg]))
        
        

        # self.model.fit(self.X_train_normalized, self.Y_train)
        
        self.train()
        
        # trainer = Trainer(self.training_output_path, verbose=verbose)
        #
        # trainer.train(
        #     x_train=self.X_train_normalized.data,
        #     y_train=self.Y_train.data,
        #     x_test=None,
        #     y_test=None,
        #     model=self.model,
        #     force=True,
        #     use_callbacks=True,
        #     verbose=int(verbose),
        #     output_path=self.training_output_path,
        #     ** self.training_params,
        # )

    def train(
            self,
            use_callbacks=False,
            batch_size=32,
            epochs=10,
            output_path=None,
            learning_rate=0.01,
            es_patience=10,
            lr_patience=9,
            lr_factor=0.5,
    ):
        self.pickle_file_path = utils.smartpath(self.training_output_path)
        self.config = PklFile(self.pickle_file_path)

        defaults = {
            'name': self.training_output_path,
            'trained': False,
            'model_json': '',
            'batch_size': [],
            'epoch_splits': [],
            'metrics': {},
        }

        for k, v in defaults.items():
            if k not in self.config:
                self.config[k] = v
        
        # callbacks = None
        #
        # if use_callbacks:
        #     print("Saving training history in: ", (output_path + ".csv"))
        #
        #     weights_file_path = self.pickle_file_path.replace(".pkl", "_weights.h5")
        #
        #     callbacks = [
        #         EarlyStopping(monitor='val_loss', patience=es_patience, verbose=0),
        #         ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, verbose=0),
        #         CSVLogger((output_path + ".csv"), append=True),
        #         TerminateOnNaN(),
        #         ModelCheckpoint(weights_file_path, monitor='val_loss', verbose=True, save_best_only=True,
        #                         save_weights_only=True, mode='min')
        #     ]
    
        model = self.model
    
        self.start_timestamp = datetime.now()
    
        previous_epochs = self.config['epoch_splits']
    
        master_epoch_n = sum(previous_epochs)
        finished_epoch_n = master_epoch_n + epochs
    
        # history = odict()
    
        # if not use_callbacks:
        #     for epoch in range(epochs):
        #         print("TRAINING EPOCH ", master_epoch_n, "/", finished_epoch_n)
        #         self.model.fit(self.X_train_normalized, self.Y_train)
        #         master_epoch_n += 1
        #
        # else:
    
        self.model.fit(self.X_train_normalized, self.Y_train)
        master_epoch_n += epochs
    
        #
    
        print("trained epochs!")
    
        self.config['trained'] = True
    
        # load the last model
        
        self.end_timestamp = datetime.now()
    
        print("finished epoch N: {}".format(finished_epoch_n))
        print("model saved")
    
        self.config['time'] = str(self.end_timestamp - self.start_timestamp)
        self.config['epochs'] = epochs
        self.config['batch_size'] = self.config['batch_size'] + [batch_size, ] * finished_epoch_n
        self.config['epoch_splits'] = previous_epochs
    
        del self.config
        
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
