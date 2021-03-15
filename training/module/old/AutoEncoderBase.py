from module.Logger import Logger

import keras

class AutoEncoderBase(Logger):

    def __init__(self, name="autoencoder", verbose=True):
        Logger.__init__(self)
        self._LOG_PREFIX = "base_autoencoder :: "
        self.VERBOSE = verbose
        self.name = name
        self.layers = []

    def __str__(self):
        s = self.log('Current Structure:', True)
        for layer in self.layers:
            s += self.log("{0}: {1} nodes {2}".format(layer[0], layer[1], layer[2:]), True)
        return s

    def __repr__(self):
        return str(self)

    def add(self, nodes, activation='relu'):
        layer = {
            "units": nodes,
            "activation": activation,
        }
        
        self.layers.append(layer)

    def build(self, optimizer='adam', loss='mse'):

        assert len(self.layers) >= 3, "need to have input, bottleneck, output!"
    
        input_size = self.layers[0]["units"]
        
        keras_input_layer = keras.layers.Input(shape=(input_size,))
        encoded = keras_input_layer
        for layer in self.layers[1:-1]:
            encoded = keras.layers.Dense(**layer)(encoded)
        encoded = keras.layers.Dense(units=input_size)(encoded)
        
        autoencoder = keras.Model(keras_input_layer, encoded)
        autoencoder.compile(optimizer, loss, metrics=['accuracy'])

        return autoencoder