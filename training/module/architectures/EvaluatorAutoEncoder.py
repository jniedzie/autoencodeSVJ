import glob, os
import tensorflow as tf
import keras
from keras.models import model_from_json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pandas as pd
import pickle

from module.DataLoader import DataLoader
from module.architectures.DenseTiedLayer import DenseTiedLayer


class EvaluatorAutoEncoder:
    def __init__(self, input_path, custom_objects):
        
        self.custom_objects = custom_objects
        
        self.signal_dict = {}
        for path in glob.glob(input_path):
            key = path.split("/")[-3]
            self.signal_dict[key] = path

        self.qcd_data = None
        self.qcd_test_weights = None
    
    def get_weights(self, summary):
        model = self.__load_model(summary)
        return model.get_weights()

    def get_qcd_data(self, summary, data_processor, data_loader, normalize=False, test_data_only=True):
        
        self.qcd_data = data_loader.get_data(data_path=summary.qcd_path,
                                             name="QCD",
                                             weights_path=summary.qcd_weights_path)
        self.qcd_weights = data_loader.weights["QCD"]
        
        if test_data_only:
            (_, _, self.qcd_data, _, _, self.qcd_test_weights) = data_processor.split_to_train_validate_test(self.qcd_data,
                                                                                                             weights=self.qcd_weights)

        if normalize:
            self.qcd_data = data_processor.normalize(data_table=self.qcd_data,
                                            normalization_type=summary.norm_type,
                                            norm_args=summary.norm_args)
        
        return self.qcd_data

    def get_qcd_weights(self, test_data_only=True):
        """
        Parameters
        ----------
        test_data_only: bool, optional
            Get weights for test part of the data only
        
        Returns
        -------
        dict[str, float]
            weights

        """
    
        if test_data_only:
            return self.qcd_test_weights
    
        return self.qcd_weights
    
    def get_signal_data(self, name, path, summary, data_processor, data_loader, normalize=False, scaler=None, test_data_only=True):
        
        data = data_loader.get_data(data_path=path, name=name)
        
        if test_data_only:
            (_, _, data, _, _, _) = data_processor.split_to_train_validate_test(data)
        
        if normalize:
            data = data_processor.normalize(data_table=data,
                                            normalization_type=summary.norm_type,
                                            norm_args=summary.norm_args,
                                            scaler=scaler)
        
        return data
        
    def get_reconstruction(self, input_data, summary, data_processor, scaler):
    
        input_data_normed = data_processor.normalize(data_table=input_data,
                                                     normalization_type=summary.norm_type,
                                                     norm_args=summary.norm_args,
                                                     scaler=scaler)

        model = self.__load_model(summary)
        
        reconstructed = pd.DataFrame(model.predict(input_data_normed.data),
                                     columns=input_data_normed.columns,
                                     index=input_data_normed.index,
                                     dtype="float64")

        descaler = input_data.scaler if scaler is None else scaler

        reconstructed_denormed = data_processor.normalize(data_table=reconstructed,
                                                          normalization_type=summary.norm_type,
                                                          norm_args=summary.norm_args,
                                                          scaler=descaler,
                                                          inverse=True)
        
        return reconstructed_denormed

    def get_latent_space_values(self, input_data, summary, data_processor, scaler):

        input_data_normed = data_processor.normalize(data_table=input_data,
                                                     normalization_type=summary.norm_type,
                                                     norm_args=summary.norm_args,
                                                     scaler=scaler)

        model = self.__load_model(summary)

        n_layers = len(summary.intermediate_architecture) + 2
        encoder = tf.keras.Model(model.input, model.layers[-n_layers].output)
        print("\n\nEncoder:")
        encoder.summary()

        predicted_data = encoder.predict(input_data_normed.data)
        latent_values = pd.DataFrame(predicted_data, dtype="float64")

        return latent_values

        
    def get_error(self, input_data, summary, data_processor, scaler):
        recon = self.get_reconstruction(input_data, summary, data_processor, scaler)
        
        print("Calculating error using loss: ", summary.loss)
        
        func = getattr(keras.losses, summary.loss)
        losses = keras.backend.eval(func(input_data.data, recon))
        
        return losses.tolist()
        
    def get_aucs(self, summary, AUCs_path, filename, data_processor, data_loader):
        
        auc_path = AUCs_path + "/" + filename
    
        if os.path.exists(auc_path):
            print("File :", auc_path, "\talready exists. Skipping...")
            return None
        else:
            print("Preparing: ", auc_path)
    
        tf.compat.v1.reset_default_graph()
    
        model = self.__load_model(summary)

        if model is None:
            print("Model is None")
            return None

        print("using summary: ", summary)
        print("AUCs path:", auc_path)
        print("filename: ", filename)
    
        aucs = self.__get_aucs(summary=summary,
                               data_processor=data_processor,
                               data_loader=data_loader,
                               model=model,
                               loss_function=summary.loss
                               )
        
        print("aucs: ", aucs)
        
        append = False
        write_header = True
        
        return (aucs, auc_path, append, write_header)

    def draw_roc_curves(self, summary, data_processor, data_loader, ax, colors, signals, test_key="qcd", **kwargs):

        normed = {
            test_key: self.get_qcd_data(summary, data_processor, data_loader, normalize=True, test_data_only=True)
        }

        for name, path in signals.items():
            normed[name] = self.get_signal_data(name, path, summary, data_processor, data_loader,
                                                normalize=True, scaler = normed[test_key].scaler)
        
        errors = {}
        model = self.__load_model(summary)

        for key, data in normed.items():
            recon = pd.DataFrame(model.predict(data.data), columns=data.columns, index=data.index, dtype="float64")
            func = getattr(keras.losses, summary.loss)
            losses = keras.backend.eval(func(data.data, recon))
            errors[key] = losses.tolist()

        signals_errors = {}
        
        for signal in signals:
            signals_errors[signal] = errors[signal]

        qcd_errors = errors[test_key]

        self.__roc_auc_plot(qcd_errors, signals_errors, ax, colors)

    def __roc_auc_plot(self, background_errors, signal_errs, ax, colors):
    
        background_labels = [0] * len(background_errors)
        
        i = 0
        
        for name, signal_err in signal_errs.items():
            pred = signal_err + background_errors
            true = [1] * len(signal_err)
            true = true + background_labels
            
            roc = roc_curve(y_true=true, y_score=pred)
            auc = roc_auc_score(y_true=true, y_score=pred)
            
            ax.plot(roc[0], roc[1], "-", c=colors[i % len(colors)], label='{}, AUC {:.4f}'.format(name, auc))

            i += 1
        
    def __load_model(self, summary):
    
        model_path = summary.training_output_path + ".tf"

        try:
            print("Trying to read model with keras load_model with .tf extension")
            model = tf.keras.models.load_model(model_path, custom_objects=self.custom_objects)
        except:
            print("Failed reading model with keras load_model with .tf extension")
            model_path = model_path.replace(".tf", ".h5")

            try:
                print("Trying to read model with keras load_model with .h5 extension")
                model = tf.keras.models.load_model(model_path, custom_objects=self.custom_objects)
            except:
                print("Failed reading model with keras load_model with .h5 extensiion")
                model_path = model_path.replace(".h5", ".pkl")
                try:
                    print("Trying to read model with model_from_json")
                    model_file = open(model_path, 'rb')
                    model = model_from_json(pickle.load(model_file), custom_objects=self.custom_objects)
                except IOError:
                    print("Failed reading model with model_from_json")
                    return None

        print("Model loaded")

        try:
            print("Trying to load weights from h5 file")
            model.load_weights(summary.training_output_path + "_weights.h5")
        except:
            print("Failed")
            try:
                print("Trying to load weights from tf file")
                model.load_weights(summary.training_output_path + "_weights.tf")
            except:
                print("Failed")

        print("Weights loaded")

        return model

    def __update_signals_efp_base(self, summary, test_key="qcd"):
        background_path_components = summary.qcd_path.split("/")
        efp_base_path = None
    
        for element in background_path_components:
            if "base" in element:
                efp_base_path = element
                break
    
        for name, path in self.signal_dict.items():
            if name == test_key: continue
        
            path_components = path.split("/")
            new_path = ""
        
            for element in path_components:
                if "base" in element:
                    new_path += efp_base_path
                else:
                    new_path += element
                new_path += "/"
                
            new_path = new_path[:-1]
            
            self.signal_dict[name] = new_path
         
    def __get_normalized_data(self, summary, data_processor, data_loader, test_key="qcd"):
        normed = {
            test_key: self.get_qcd_data(summary, data_processor, data_loader, normalize=True, test_data_only=True)
        }
    
        self.__update_signals_efp_base(summary, test_key)
    
        for name, path in self.signal_dict.items():
            if name == test_key: continue
        
            normed[name] = self.get_signal_data(name=name, path=path,
                                                summary=summary,
                                                data_processor=data_processor, data_loader=data_loader,
                                                normalize=True, scaler=normed[test_key].scaler)
            
        return normed
        
    def __get_losses(self, data, model, loss_function):
        losses = {}
    
        for key, data in data.items():
            recon = pd.DataFrame(model.predict(data.data), columns=data.columns, index=data.index, dtype="float64")
            func = getattr(keras.losses, loss_function)
            loss = keras.backend.eval(func(data.data, recon))
            losses[key] = loss.tolist()
            
        return losses
        
    def __get_aucs(self, summary, data_processor, data_loader, model, loss_function, test_key='qcd'):
        """
        
        Args:
            summary:
            data_processor:
            data_loader (DataLoader):
            model:
            loss_function:
            test_key:

        Returns:

        """
        
        normed = self.__get_normalized_data(summary, data_processor, data_loader, test_key)
        
        errors = self.__get_losses(normed, model, loss_function)

        background_errors = errors[test_key]
        background_labels = [0] * len(background_errors)
        background_weights = self.get_qcd_weights(test_data_only=True)
    
        if background_weights is None:
            background_weights = [1] * len(background_errors)
        else:
            background_weights = background_weights.tolist()
    
        aucs = []
        for name, signal_err in errors.items():
            if name == test_key: continue
        
            pred = signal_err + background_errors
            true = [1] * len(signal_err)
            true = true + background_labels
        
            weights = [1] * len(signal_err)
            weights = weights + background_weights
        
            auc = roc_auc_score(y_true=true, y_score=pred, sample_weight=weights)
        
            signal_components = name.split("_")
            mass_index = [i for i, s in enumerate(signal_components) if 'GeV' in s][0]
            mass = int(signal_components[mass_index].strip("GeV"))
            rinv = float(signal_components[mass_index + 1])
        
            aucs.append({"mass": mass, "rinv": rinv, "auc": auc})
    
        return aucs