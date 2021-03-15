import module.utils as utils
import module.SummaryProcessor as summaryProcessor
from module.DataProcessor import DataProcessor
from module.DataLoader import DataLoader
import numpy as np
import datetime
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint, CSVLogger
from module.PklFile import PklFile


class TrainerAutoEncoder:
    
    def __init__(self,
                 qcd_path,
                 training_params,
                 bottleneck_size,
                 output_file_name,
                 training_output_path,
                 EFP_base=None,
                 intermediate_architecture=(30, 30),
                 test_data_fraction=0.15,
                 validation_data_fraction=0.15,
                 norm_type="",
                 norm_args=None,
                 hlf_to_drop=None,
                 verbose=True
                 ):
        """
        Creates auto-encoder trainer with random seed, provided training parameters and architecture.
        Loads specified data, splits them into training, validation and test samples according to
        provided arguments. Normalizes the data as specified by norm_percentile.
        High-level features specified in hlf_to_drop will not be used for training.
        """
        
        self.seed = np.random.randint(0, 99999999)
        utils.set_random_seed(self.seed)
        
        self.qcd_path = qcd_path
        self.hlf_to_drop = hlf_to_drop
        self.EFP_base = EFP_base
        
        self.training_params = training_params
        self.test_data_fraction = test_data_fraction
        self.validation_data_fraction = validation_data_fraction
        self.training_output_path = training_output_path + output_file_name
        
        self.verbose = verbose
        
        data_loader = DataLoader()
        
        # Load QCD samples
        (self.qcd, _, _, _) = data_loader.load_all_data(qcd_path, "qcd background",
                                                        include_hlf=True, include_eflow=True,
                                                        hlf_to_drop=hlf_to_drop)
        
        data_processor = DataProcessor(validation_fraction=self.validation_data_fraction,
                                       test_fraction=self.test_data_fraction,
                                       seed=self.seed)
        
        (train_data,
         validation_data,
         test_data, _, _) = data_processor.split_to_train_validate_test(data_table=self.qcd)
        
        train_data.output_file_prefix = "qcd training data"
        validation_data.output_file_prefix = "qcd validation data"
        
        # Normalize the input
        self.norm_type = norm_type
        self.norm_args = norm_args
        
        print("Trainer scaler args: ", self.norm_args)
        
        self.train_data_normalized = data_processor.normalize(data_table=train_data,
                                                              normalization_type=self.norm_type,
                                                              norm_args=self.norm_args,
                                                              )
        
        self.validation_data_normalized = data_processor.normalize(data_table=validation_data,
                                                                   normalization_type=self.norm_type,
                                                                   norm_args=self.norm_args
                                                                   )
        
        # Build the model
        self.input_size = len(self.qcd.columns)
        self.intermediate_architecture = intermediate_architecture
        self.bottleneck_size = bottleneck_size
        self.model = self.__get_auto_encoder_model()

    def train(self):
        """
        Runs the training on data loaded and prepared in the constructor, according to training params
        specified in the constructor
        """
        
        self.pickle_file_path = utils.smartpath(self.training_output_path) + ".pkl"
        self.config = PklFile(self.pickle_file_path)
    
        print("\n\nTraining the model")
        print("Filename: ", self.training_output_path)
        print("Number of training samples: ", len(self.train_data_normalized.data))
        print("Number of validation samples: ", len(self.validation_data_normalized.data))
    
        if self.verbose:
            self.model.summary()
            print("\nTraining params:")
            for arg in self.training_params:
                print((arg, ":", self.training_params[arg]))
    
        self.start_timestamp = datetime.datetime.now()
    
        callbacks = self.__get_callbacks()

        self.model.fit(
            x=self.train_data_normalized.data,
            y=self.train_data_normalized.data,
            validation_data=(self.validation_data_normalized.data, self.validation_data_normalized.data),
            epochs=self.training_params["epochs"],
            batch_size=self.training_params["batch_size"],
            verbose=self.verbose,
            callbacks=callbacks
        )
        
        self.config['trained'] = True
        self.config['model_json'] = str(self.model.to_json())

        print("\ntrained {} epochs!", self.training_params["epochs"], "\n")
        print("model saved")
        self.end_timestamp = datetime.datetime.now()
        print("Training executed in: ", (self.end_timestamp - self.start_timestamp), " s")

    def save_summary(self, path):
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
            'target_dim': self.bottleneck_size,
            'input_dim': self.input_size,
            'arch': self.__get_architecture_summary(),
            'seed': self.seed,
            'start_time': str(self.start_timestamp),
            'end_time': str(self.end_timestamp),
        }
        summaryProcessor.dump_summary_json(self.training_params, summary_dict, output_path=path)

    def __get_architecture_summary(self):
        """
        Returns a tuple with number of nodes in each consecutive layer of the auto-encoder
        """
        arch = (self.input_size,) + self.intermediate_architecture
        arch += (self.bottleneck_size,)
        arch += tuple(reversed(self.intermediate_architecture)) + (self.input_size,)
        return arch

    def __get_auto_encoder_model(self):
        """
        Builds an auto-encoder model as specified in object's fields: input_size,
        intermediate_architecture and bottleneck_size
        """
        
        input_layer = keras.layers.Input(shape=(self.input_size,))
        layers = input_layer
    
        for elt in self.intermediate_architecture:
            layers = keras.layers.Dense(units=elt, activation= "relu")(layers)

        layers = keras.layers.Dense(units=self.bottleneck_size, activation="relu")(layers)
        
        for elt in reversed(self.intermediate_architecture):
            layers = keras.layers.Dense(units=elt, activation= "relu")(layers)

        layers = keras.layers.Dense(units=self.input_size, activation="linear")(layers)

        autoencoder = keras.Model(input_layer, layers)
        autoencoder.compile(optimizer=self.training_params["optimizer"],
                            loss=self.training_params["loss"],
                            metrics=self.training_params["metric"])

        return autoencoder
    
    def __get_callbacks(self):
        callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=self.training_params["es_patience"],
                          verbose=self.verbose),
            ReduceLROnPlateau(monitor='val_loss',
                              factor=self.training_params["lr_factor"],
                              patience=self.training_params["lr_patience"],
                              verbose=self.verbose),
            CSVLogger(filename=(self.training_output_path + ".csv"),
                      append=True),
            TerminateOnNaN(),
            ModelCheckpoint(self.training_output_path + "_weights.h5",
                            monitor='val_loss',
                            verbose=self.verbose,
                            save_best_only=True,
                            save_weights_only=True,
                            mode='min')
        ]
        return callbacks