from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint, CSVLogger
from tensorflow.keras import layers
import tensorflow as tf

from module.architectures.vaeHelpers import *

class TrainerAutoEncoder:
    
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

        tf.compat.v1.disable_eager_execution()
        
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
        # self.__model = self.__get_model()
        self.__model = self.__get_model_vae()

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
        (self.qcd, _, _, _) = self.data_loader.load_all_data(self.qcd_path, "QCD")
        (self.train_data, self.validation_data, _) = self.data_processor.split_to_train_validate_test(data_table=self.qcd)

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

    def train(self):
        """
        @mandatory
        Runs the training of the previously prepared model on the normalized data
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
        @mandatory
        Add additional information to be stored in the summary file. Can return empty dict if
        no additional information is needed.
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
        
        input_layer = tf.keras.layers.Input(shape=(self.input_size,))
        ae_layers = input_layer
    
        for elt in self.training_params["intermediate_architecture"]:
            ae_layers = layers.Dense(units=elt, activation= "relu")(ae_layers)

        ae_layers = layers.Dense(units=self.training_params["bottleneck_size"], activation="relu")(ae_layers)
        
        for elt in reversed(self.training_params["intermediate_architecture"]):
            ae_layers = layers.Dense(units=elt, activation= "relu")(ae_layers)

        ae_layers = layers.Dense(units=self.input_size, activation="linear")(ae_layers)

        autoencoder = tf.keras.Model(input_layer, ae_layers)
        autoencoder.compile(optimizer=self.training_params["optimizer"],
                            loss=self.training_params["loss"],
                            metrics=self.training_params["metric"])

        return autoencoder
    
    def __get_callbacks(self):
        """
        Initializes and returns training callbacks
        """
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

    # def build_vae(self):
    #
    #     latent_dim = self.training_params["bottleneck_size"]
    #     middle_arch = self.training_params["intermediate_architecture"]
    #     loss = self.training_params["loss"]
    #
    #
    #     inputs = layers.Input(shape=(self.input_size,))
    #     last = inputs
    #     first_middle = []
    #
    #     for i, n in enumerate(middle_arch):
    #         first_middle.append(layers.Dense(n, activation='relu', name='interm_1.{}'.format(i))(last))
    #         last = first_middle[i]
    #
    #     z_mean = layers.Dense(latent_dim, name='z_mean')(last)
    #     z_log_var = layers.Dense(latent_dim, name='z_log_var')(last)
    #
    #     z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    #
    #     encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    #
    #     latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
    #     last = latent_inputs
    #     second_middle = []
    #     for i, n in enumerate(reversed(middle_arch)):
    #         second_middle.append(layers.Dense(n, activation='relu', name='interm_2.{}'.format(i))(last))
    #         last = second_middle[i]
    #     outputs = layers.Dense(self.input_size, activation='linear', name='outputs')(last)
    #
    #     decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')
    #
    #     output = decoder(encoder(inputs)[2])
    #     vae = tf.keras.Model(inputs, output)
    #
    #     reconstruction_loss = getattr(tf.keras.losses, loss)(inputs, output)
    #     kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    #     kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    #     kl_loss *= -0.5
    #
    #     vae_loss =tf.keras.backend.mean(reconstruction_loss + kl_loss)
    #
    #     vae.compile(optimizer=self.training_params["optimizer"],
    #                         loss=lambda x,y: vae_loss,
    #                         metrics=self.training_params["metric"])
    #
    #     return vae

    
    # minimal example
    def __get_model_vae(self):
    
        # encoder
        input_layer = layers.Input(shape=(self.input_size,))
        layer_1 = layers.Dense(10, activation='relu')(input_layer)
        
        z_mean = layers.Dense(10)(layer_1)
        z_log_var = layers.Dense(10)(layer_1)
        z = layers.Lambda(sampling, output_shape=(10,))([z_mean, z_log_var])
    
        encoder = tf.keras.models.Model(input_layer, [z_mean, z_log_var, z])
    
        # decoder
        decoder_input_layer = layers.Input(shape=(10,))
        layer_n1 = layers.Dense(10, activation='relu')(decoder_input_layer)
        output_layer = layers.Dense(self.input_size, activation='linear')(layer_n1)
    
        decoder = tf.keras.models.Model(decoder_input_layer, output_layer)

        vae_out = decoder(encoder(input_layer))

        autoencoder = tf.keras.Model(input_layer, vae_out)
    
       
        autoencoder.compile(optimizer=self.training_params["optimizer"],
                            loss=vae_loss(z_log_var=z_log_var, z_mean=z_mean),
                            metrics=[self.training_params["metric"]])
    
        return autoencoder
