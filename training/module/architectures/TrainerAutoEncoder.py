from module.architectures.TrainerAutoEncoderBase import TrainerAutoEncoderBase
from module.architectures.DenseTiedLayer import DenseTiedLayer

from tensorflow.keras import layers
import tensorflow as tf


class TrainerAutoEncoder(TrainerAutoEncoderBase):
    
    def __init__(self, **args):
        """
        Constructor of the specialized Trainer class.
        data_processor and data_loader fields are mandatory and will be passed, ready to be used.
        Names of the remaining arguments match keys of the "training_settings" dict from the config.
        """

        super(TrainerAutoEncoder, self).__init__(**args)
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
        super(TrainerAutoEncoder, self).train()
        
    def get_summary(self):
        """
        @mandatory
        Add additional information to be stored in the summary file. Can return empty dict if
        no additional information is needed.
        """
        return super(TrainerAutoEncoder, self).get_summary()

    def __get_model(self):
        """
        Builds an auto-encoder model as specified in object's fields: input_size,
        intermediate_architecture and bottleneck_size
        """

        input_layer = layers.Input(shape=(self.input_size,))
        ae_layers = input_layer
        activation = self.training_params["activation"]
        tied_weights = self.training_params["tied_weights"]

        encoder_layers = []

        for elt in self.training_params["intermediate_architecture"]:
            layer = layers.Dense(units=elt, activation=activation)
            ae_layers = layer(ae_layers)
            encoder_layers.append(layer)

        bottleneck_layer = layers.Dense(units=self.training_params["bottleneck_size"], activation=activation)
        ae_layers = bottleneck_layer(ae_layers)
        encoder_layers.append(bottleneck_layer)

        for i, elt in enumerate(reversed(self.training_params["intermediate_architecture"])):
            if tied_weights:
                layer = DenseTiedLayer(units=elt, activation=activation, tied_to=encoder_layers[-(i + 1)])
            else:
                layer = layers.Dense(units=elt, activation=activation)
            ae_layers = layer(ae_layers)

        if tied_weights:
            output_layer = DenseTiedLayer(units=self.input_size,
                                          activation=self.training_params["output_activation"],
                                          tied_to=encoder_layers[0])
        else:
            output_layer = layers.Dense(units=self.input_size,
                                        activation=self.training_params["output_activation"])

        ae_layers = output_layer(ae_layers)

        autoencoder = tf.keras.Model(input_layer, ae_layers)
        autoencoder.compile(optimizer=self.training_params["optimizer"],
                            loss=self.training_params["loss"],
                            metrics=[self.training_params["metric"]])

        return autoencoder
