from module.architectures.TrainerAutoEncoderBase import TrainerAutoEncoderBase

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
            ae_layers = layers.Dense(units=elt, activation= "relu")(ae_layers)

        ae_layers = layers.Dense(units=self.training_params["bottleneck_size"], activation="relu")(ae_layers)
        
        for elt in reversed(self.training_params["intermediate_architecture"]):
            ae_layers = layers.Dense(units=elt, activation= "relu")(ae_layers)

        ae_layers = layers.Dense(units=self.input_size, activation="linear")(ae_layers)

        autoencoder = tf.keras.Model(input_layer, ae_layers)
        autoencoder.compile(optimizer=self.training_params["optimizer"],
                            loss=self.training_params["loss"],
                            metrics=[self.training_params["metric"]])

        return autoencoder
