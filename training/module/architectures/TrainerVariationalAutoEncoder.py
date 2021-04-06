from module.architectures.TrainerAutoEncoderBase import TrainerAutoEncoderBase
from module.architectures.vaeHelpers import *

from tensorflow.keras import layers


class TrainerVariationalAutoEncoder(TrainerAutoEncoderBase):
    
    def __init__(self, **args):
        """
        Constructor of the specialized Trainer class.
        data_processor and data_loader fields are mandatory and will be passed, ready to be used.
        Names of the remaining arguments match keys of the "training_settings" dict from the config.
        """
        tf.compat.v1.disable_eager_execution()
        super(TrainerVariationalAutoEncoder, self).__init__(**args)
        self._model = self.__get_model()
    
    @property
    def model(self):
        """
        @mandatory
        Property that should return the model
        """
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
    
    def train(self):
        """
        @mandatory
        Runs the training of the previously prepared model on the normalized data
        """
        super(TrainerVariationalAutoEncoder, self).train()
    
    def get_summary(self):
        """
        @mandatory
        Add additional information to be stored in the summary file. Can return empty dict if
        no additional information is needed.
        """
        return super(TrainerVariationalAutoEncoder, self).get_summary()

    def __get_model(self):

        latent_dim = self.training_params["bottleneck_size"]
        middle_arch = self.training_params["intermediate_architecture"]

        # encoder
        input_layer = layers.Input(shape=(self.input_size,))
        last = input_layer
        first_middle = []

        for i, n in enumerate(middle_arch):
            first_middle.append(layers.Dense(n, activation='relu')(last))
            last = first_middle[i]

        z_mean = layers.Dense(latent_dim)(last)
        z_log_var = layers.Dense(latent_dim)(last)
        z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        encoder = tf.keras.models.Model(input_layer, [z_mean, z_log_var, z])

        # decoder
        decoder_input_layer = layers.Input(shape=(latent_dim,))
        last = decoder_input_layer
        second_middle = []
        for i, n in enumerate(reversed(middle_arch)):
            second_middle.append(layers.Dense(n, activation='relu')(last))
            last = second_middle[i]
        output_layer = layers.Dense(self.input_size, activation='linear')(last)

        decoder = tf.keras.models.Model(decoder_input_layer, output_layer)

        # VAE
        vae_out = decoder(encoder(input_layer))
        model = tf.keras.Model(input_layer, vae_out)
        model.compile(optimizer=self.training_params["optimizer"],
                            loss=vae_loss(z_log_var=z_log_var, z_mean=z_mean, reco_loss=self.training_params["loss"]),
                            metrics=[self.training_params["metric"]])

        return model
