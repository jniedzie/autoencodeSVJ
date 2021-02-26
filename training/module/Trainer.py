from module.Logger import Logger
from module.PklFile import PklFile
import module.utils as utils

from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint, CSVLogger
import keras.optimizers

from collections import OrderedDict as odict
from datetime import datetime
import traceback
import os
import numpy as np

class Trainer(Logger):
    """
    Wraps training/testing/evaluation activity for a model in an h5 file saver, which keeps all
    training inputs/outputs, model performance stats, model weights, etc.
    """

    ### LIFE/DEATH

    def __init__(self, name, verbose=True):
        Logger.__init__(self)

        self._LOG_PREFIX = "train_shell :: "
        self.VERBOSE = verbose

        self.pickle_file_path = utils.smartpath(name)
        self.path = os.path.dirname(self.pickle_file_path)

        if not self.pickle_file_path.endswith(".pkl"):
            self.pickle_file_path += ".pkl"
            
        self.config = PklFile(self.pickle_file_path)
        
        defaults = {
            'name': name,
            'trained': False,
            'model_json': '',
            'batch_size': [],
            'epoch_splits': [],
            'metrics': {},
        }

        for k, v in defaults.items():
            if k not in self.config:
                self.config[k] = v

    def _throw(self, msg, exc=AttributeError):
        self._close()
        self.error(msg)
        raise exc

    def _close(self):
        try:
            del self.config
        except:
            pass

    def __del__(self):
        self._close()

    ### ACTUAL WRAPPERS

    def load_model(self, model=None, force=False):
        
        weights_file_path = self.pickle_file_path.replace(".pkl", "_weights.h5")
        # if already trained
        if self.config['trained']:
            if self.config['model_json']:
                if model is None:
                    model = model_from_json(self.config['model_json'])
                    model.load_weights(weights_file_path)
                    self.log("using saved model")
                else:
                    if not force:
                        model = model_from_json(self.config['model_json'])
                        model.load_weights(weights_file_path)
                        self.error("IGNORING PASSED PARAMETER 'model'")
                        self.log("using saved model")
                    else:
                        if isinstance(model, str):
                            model = self.load_model(model)
                        self.log("using model passed as function argument")
            else:
                if model is None:
                    self._throw("no model passed, and saved model not found!")
                if isinstance(model, str):
                    model = self.load_model(model)
                self.log("using model passed as function argument")
        
        if model is None:
            self._throw("no model passed and no saved model found!!")
            
        return model

    def train(
        self,
        x_train,
        y_train=None,
        x_test=None,
        y_test=None,
        model=None,
        force=False,
        use_callbacks=False,
        verbose=1,
        output_path=None,
        
        batch_size=32,
        loss=None,
        optimizer=None,
        epochs=10,
        learning_rate=0.01,
        es_patience=10,
        lr_patience=9,
        lr_factor=0.5,
    ):
        callbacks = None

        weights_file_path = self.pickle_file_path.replace(".pkl", "_weights.h5")
        if use_callbacks:
            
            print("Saving training history in: ", (output_path+".csv"))
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=es_patience, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, verbose=0),
                CSVLogger((output_path+".csv"), append=True),
                TerminateOnNaN(),
                ModelCheckpoint(weights_file_path, monitor='val_loss', verbose=self.VERBOSE, save_best_only=True, save_weights_only=True, mode='min')
            ]
        
        model = self.load_model(model, force)

        if optimizer is None:
            if hasattr(model, "optimizer"):
                optimizer = model.optimizer
            else:
                optimizer = getattr(keras.optimizers, "adam")(lr=learning_rate)
        elif isinstance(optimizer, str):
            optimizer = getattr(keras.optimizers, "Adam")(lr=learning_rate)

        loss_weights = None
        if loss is None:
            if hasattr(model, "loss"):
                loss = model.loss
                loss_weights = model.loss_weights
            else:
                loss = "mse"
                loss_weights = [1.]
        
        if loss_weights is None:
            if isinstance(loss, list):
                loss_weights = [1. for i in range(len(loss))]
            else:
                loss = [loss]
                loss_weights = [1.]

        metrics = model.metrics or []

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)

        start = datetime.now()

        if y_train is None:
            y_train = x_train
        
        if x_test is None:
            x_test = np.zeros((0,) + x_train.shape[1:])

        if y_test is None:
            y_test = x_test.copy()

        previous_epochs = self.config['epoch_splits']

        master_epoch_n = sum(previous_epochs)
        finished_epoch_n = master_epoch_n + epochs
    
        history = odict()

        if not use_callbacks:
            try:
                for epoch in range(epochs):
                    self.log("TRAINING EPOCH {}/{}".format(master_epoch_n, finished_epoch_n))
                    
                    nhistory = model.fit(
                        x=x_train,
                        y=y_train,
                        validation_data=(x_test, y_test),
                        initial_epoch=master_epoch_n,
                        epochs=master_epoch_n + 1,
                        verbose=verbose,
                        callbacks=callbacks,
                        batch_size=batch_size,
                    ).history

                    if epoch == 0:
                        for metric in nhistory:
                            history[metric] = []

                    for metric in nhistory:
                        history[metric].append([master_epoch_n, nhistory[metric][0]])

                    master_epoch_n += 1

            except:
                self.error(traceback.format_exc())
                if all([len(v) == 0 for v in history]):
                    self._throw("quitting")
                self.log("saving to path " + weights_file_path)
                model.save_weights(weights_file_path)

            if len(list(history.values())) == 0:
                n_epochs_finished = 0
            else:
                n_epochs_finished = min(list(map(len, list(history.values()))))

        else:
            nhistory = model.fit(
                x=x_train,
                y=y_train,
                validation_data=(x_test, y_test),
                initial_epoch=master_epoch_n,
                epochs=master_epoch_n + epochs,
                verbose=verbose,
                callbacks=callbacks,
                batch_size=batch_size,
            ).history

            for metric in nhistory:
                history[metric] = []
                for i,value in enumerate(nhistory[metric]):
                    history[metric].append([master_epoch_n + i, value])
            master_epoch_n += epochs
            n_epochs_finished = min(list(map(len, list(history.values()))))

        # self.log("EPOCH N: {}, {}".format(master_epoch_n, epochs))
        self.log("")
        self.log("trained {} epochs!".format(n_epochs_finished))
        self.log("")

        js = model.to_json()
        self.config['trained'] = True 
        self.config['model_json'] = str(js)
    
        # load the last model
        hvalues = [hv[:n_epochs_finished] for hv in list(history.values())]
        hkeys = list(history.keys())

        nmetrics = self.config['metrics'].copy()
        for key,value in zip(hkeys, hvalues):
            if key in nmetrics:
                nmetrics[key] = np.concatenate([nmetrics[key], value])
            else:
                nmetrics[key] = np.asarray(value)

        self.config['metrics'] = nmetrics

        previous_epochs.append(n_epochs_finished)
        finished_epoch_n = sum(previous_epochs)

        end = datetime.now()

        self.log("finished epoch N: {}".format(finished_epoch_n))
        self.log("model saved")

        self.config['time'] = str(end - start)
        self.config['epochs'] = epochs
        self.config['batch_size'] = self.config['batch_size'] + [batch_size,]*n_epochs_finished
        self.config['epoch_splits'] = previous_epochs
    
        return model
