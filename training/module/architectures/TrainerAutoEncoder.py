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

        input_layer = tf.keras.layers.Input(shape=(self.input_size,))
        ae_layers = input_layer

        for elt in self.training_params["intermediate_architecture"]:
            ae_layers = layers.Dense(units=elt, activation=self.training_params["activation"])(ae_layers)

        ae_layers = layers.Dense(units=self.training_params["bottleneck_size"],
                                 activation=self.training_params["activation"])(ae_layers)

        for elt in reversed(self.training_params["intermediate_architecture"]):
            ae_layers = layers.Dense(units=elt, activation=self.training_params["activation"])(ae_layers)

        ae_layers = layers.Dense(units=self.input_size,
                                 activation=self.training_params["output_activation"])(ae_layers)

        autoencoder = tf.keras.Model(input_layer, ae_layers)
        autoencoder.compile(optimizer=self.training_params["optimizer"],
                            loss=self.training_params["loss"],
                            metrics=[self.training_params["metric"]])

        return autoencoder
    
    # tied weights
    # def __get_model(self):
    #     """
    #     Builds an auto-encoder model as specified in object's fields: input_size,
    #     intermediate_architecture and bottleneck_size
    #     """
    #
    #     input_layer = tf.keras.layers.Input(shape=(self.input_size,))
    #     ae_layers = input_layer
    #
    #     encoder_layers = []
    #
    #     for elt in self.training_params["intermediate_architecture"]:
    #         layer = layers.Dense(units=elt, activation="relu")
    #         ae_layers = layer(ae_layers)
    #         encoder_layers.append(layer)
    #
    #     bottleneck_layer = layers.Dense(units=self.training_params["bottleneck_size"], activation="relu")
    #     ae_layers = bottleneck_layer(ae_layers)
    #     encoder_layers.append(bottleneck_layer)
    #
    #     decoder_layers = []
    #
    #     for i, elt in enumerate(reversed(self.training_params["intermediate_architecture"])):
    #         layer = layers.Dense(units=elt, activation="relu", use_bias=False)
    #         decoder_layers.append(layer)
    #         ae_layers = layer(ae_layers)
    #
    #     output_layer = layers.Dense(units=self.input_size, activation="linear", use_bias=False)
    #     ae_layers = output_layer(ae_layers)
    #     decoder_layers.append(output_layer)
    #
    #     for i in range(len(decoder_layers)):
    #         print("i: ", i)
    #
    #         layer = decoder_layers[-(i+1)]
    #         layer.set_weights([encoder_layers[i].get_weights()[0].T])
    #
    #         # if i==0:
    #         #     layer.set_weights([encoder_layers[i].get_weights()[0].T, layer.get_weights()[1]])
    #         # else:
    #         #     layer.set_weights([encoder_layers[i].get_weights()[0].T, encoder_layers[i-1].get_weights()[1]])
    #         #
    #
    #     print("\n\nencoder layers:")
    #     for i, layer in enumerate(encoder_layers):
    #         print("i: ", i)
    #         print("weights: ", layer.get_weights())
    #
    #     print("\n\ndecoder layers:")
    #     for i, layer in enumerate(decoder_layers):
    #         print("i: ", i)
    #         print("weights: ", layer.get_weights())
    #
    #
    #
    #     autoencoder = tf.keras.Model(input_layer, ae_layers)
    #     autoencoder.compile(optimizer=self.training_params["optimizer"],
    #                         loss=self.training_params["loss"],
    #                         metrics=[self.training_params["metric"]])
    #
    #     return autoencoder

    # tied weights with custom layer
    # def __get_model(self):
    #     """
    #     Builds an auto-encoder model as specified in object's fields: input_size,
    #     intermediate_architecture and bottleneck_size
    #     """
    #
    #     input_layer = layers.Input(shape=(self.input_size,))
    #     ae_layers = input_layer
    #
    #     encoder_layers = []
    #
    #     for elt in self.training_params["intermediate_architecture"]:
    #         layer = layers.Dense(units=elt, activation="relu")
    #         ae_layers = layer(ae_layers)
    #         encoder_layers.append(layer)
    #
    #     bottleneck_layer = layers.Dense(units=self.training_params["bottleneck_size"], activation="relu")
    #     ae_layers = bottleneck_layer(ae_layers)
    #     encoder_layers.append(bottleneck_layer)
    #
    #
    #     for i, elt in enumerate(reversed(self.training_params["intermediate_architecture"])):
    #         encoder_layer = encoder_layers[-(i + 1)]
    #         layer = DenseTiedLayer(units=elt, activation="relu", tied_to=encoder_layer)
    #         ae_layers = layer(ae_layers)
    #
    #     output_layer = DenseTiedLayer(units=self.input_size, activation="linear", tied_to=encoder_layers[0])
    #     ae_layers = output_layer(ae_layers)
    #
    #
    #     autoencoder = tf.keras.Model(input_layer, ae_layers)
    #     autoencoder.compile(optimizer=self.training_params["optimizer"],
    #                         loss=self.training_params["loss"],
    #                         metrics=[self.training_params["metric"]])
    #
    #     return autoencoder


