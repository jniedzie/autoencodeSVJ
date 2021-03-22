import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint, CSVLogger


class TrainerAutoEncoder:
    
    def __init__(self,
                 qcd_path,
                 training_params,
                 training_output_path,
                 data_processor,
                 data_loader,
                 EFP_base=None,
                 norm_type="",
                 norm_args=None,
                 verbose=True
                 ):
        """
        Creates auto-encoder trainer with random seed, provided training parameters and architecture.
        Loads specified data, splits them into training, validation and test samples according to
        provided arguments. Normalizes the data as specified by norm_percentile.
        High-level features specified in hlf_to_drop will not be used for training.
        """
        self.qcd_path = qcd_path
        self.EFP_base = EFP_base
        
        self.training_params = training_params
        self.training_output_path = training_output_path
        
        self.data_processor = data_processor
        self.data_loader = data_loader
        
        self.verbose = verbose

        # Load and split the data
        self.__load_data()
        
        # Normalize the input
        self.norm_type = norm_type
        self.norm_args = norm_args
        self.__normalize_data()
        
        # Build the model
        self.input_size = len(self.qcd.columns)
        self.__model = self.__get_model()

    @property
    def model(self):
        return self.__model

    def __load_data(self):
        
        # Load QCD samples
        (self.qcd, _, _, _) = self.data_loader.load_all_data(self.qcd_path, "QCD")
        (self.train_data, self.validation_data, _) = self.data_processor.split_to_train_validate_test(data_table=self.qcd)

    def __normalize_data(self):
        print("Trainer scaler: ", self.norm_type)
        print("Trainer scaler args: ", self.norm_args)
    
        self.train_data_normalized = self.data_processor.normalize(data_table=self.train_data,
                                                                   normalization_type=self.norm_type,
                                                                   norm_args=self.norm_args
                                                                   )
    
        self.validation_data_normalized = self.data_processor.normalize(data_table=self.validation_data,
                                                                        normalization_type=self.norm_type,
                                                                        norm_args=self.norm_args
                                                                        )

    def train(self):
        """
        Runs the training on data loaded and prepared in the constructor, according to training params
        specified in the constructor
        """
        
        print("Filename: ", self.training_output_path)
        print("Number of training samples: ", len(self.train_data_normalized.data))
        print("Number of validation samples: ", len(self.validation_data_normalized.data))
    
        if self.verbose:
            self.__model.summary()
            print("\nTraining params:")
            for arg in self.training_params:
                print((arg, ":", self.training_params[arg]))
    
        callbacks = self.__get_callbacks()

        self.__model.fit(
            x=self.train_data_normalized.data,
            y=self.train_data_normalized.data,
            validation_data=(self.validation_data_normalized.data, self.validation_data_normalized.data),
            epochs=self.training_params["epochs"],
            batch_size=self.training_params["batch_size"],
            verbose=self.verbose,
            callbacks=callbacks
        )
        
        print("\ntrained {} epochs!", self.training_params["epochs"], "\n")

    def get_summary(self):
        """
        Dumps summary of the most recent training to a summary file.
        """
        summary_dict = {
            'training_output_path': self.training_output_path,
            'qcd_path': self.qcd_path,
            'hlf': True,
            'eflow': True,
            'eflow_base': self.EFP_base,
            'norm_type': self.norm_type,
            'norm_args': self.norm_args,
            'input_dim': self.input_size,
            'arch': self.__get_architecture_summary(),
        }
        
        summary_dict = {**summary_dict, **self.training_params}
        
        return summary_dict

    def __get_architecture_summary(self):
        """
        Returns a tuple with number of nodes in each consecutive layer of the auto-encoder
        """
        arch = (self.input_size,) + self.training_params["intermediate_architecture"]
        arch += (self.training_params["bottleneck_size"],)
        arch += tuple(reversed(self.training_params["intermediate_architecture"])) + (self.input_size,)
        return arch

    def __get_model(self):
        """
        Builds an auto-encoder model as specified in object's fields: input_size,
        intermediate_architecture and bottleneck_size
        """
        
        input_layer = keras.layers.Input(shape=(self.input_size,))
        layers = input_layer
    
        for elt in self.training_params["intermediate_architecture"]:
            layers = keras.layers.Dense(units=elt, activation= "relu")(layers)

        layers = keras.layers.Dense(units=self.training_params["bottleneck_size"], activation="relu")(layers)
        
        for elt in reversed(self.training_params["intermediate_architecture"]):
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