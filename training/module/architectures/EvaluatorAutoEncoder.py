import glob
import os
import tensorflow as tf
import keras
from keras.models import model_from_json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pandas as pd
import pickle

from module.DataLoader import DataLoader
from module.DataProcessor import *
from module.architectures.DenseTiedLayer import DenseTiedLayer


class EvaluatorAutoEncoder:
    def __init__(self, input_path, custom_objects):
        
        self.custom_objects = custom_objects
        
        self.signal_dict = {}
        for path in glob.glob(input_path):
            key = path.split("/")[-3]
            self.signal_dict[key] = path

        self.qcd_data = None
    
    def get_model_weights(self, summary):
        model = self.__load_model(summary)
        return model.get_weights()

    def get_qcd_data(self, summary, data_processor, data_loader, normalize=False, test_data_only=True):
        
        self.qcd_data = data_loader.get_data(data_path=summary.qcd_path, weights_path=summary.qcd_weights_path)
        
        if test_data_only:
            (_, _, self.qcd_data) = data_processor.split_to_train_validate_test(self.qcd_data)

        if normalize:
            self.qcd_data = DataProcessor.normalize(data=self.qcd_data,
                                                    normalization_type=summary.norm_type,
                                                    norm_args=summary.norm_args)
        
        return self.qcd_data
    
    def get_signal_data(self, path, summary, data_processor, data_loader, normalize=False, scaler=None,
                        test_data_only=True, use_qcd_weights=False):
        
        weights_path = summary.qcd_weights_path if use_qcd_weights else None
        
        data = data_loader.get_data(data_path=path, weights_path=weights_path)
        
        if test_data_only:
            (_, _, data) = data_processor.split_to_train_validate_test(data)
        
        if normalize:
            data = DataProcessor.normalize(data=data,
                                           normalization_type=summary.norm_type,
                                           norm_args=summary.norm_args,
                                           scaler=scaler)
        
        return data
        
    def get_reconstruction(self, input_data, summary, scaler, model=None):
    
        input_data_normed = DataProcessor.normalize(data=input_data,
                                                    normalization_type=summary.norm_type,
                                                    norm_args=summary.norm_args,
                                                    scaler=scaler)

        if model is None:
            model = self.__load_model(summary)
        
        reconstructed = pd.DataFrame(model.predict(input_data_normed.data),
                                     columns=input_data_normed.columns,
                                     index=input_data_normed.index,
                                     dtype="float64")

        reconstructed = DataTable(reconstructed)

        descaler = input_data.scaler if scaler is None else scaler

        reconstructed_denormed = DataProcessor.normalize(data=reconstructed,
                                                         normalization_type=summary.norm_type,
                                                         norm_args=summary.norm_args,
                                                         scaler=descaler,
                                                         inverse=True)
        
        return reconstructed_denormed

    def get_latent_space_values(self, input_data, summary, scaler):
    
        input_data_normed = DataProcessor.normalize(data=input_data,
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

    def get_error(self, input_data, summary, scaler, model=None):
        recon = self.get_reconstruction(input_data, summary, scaler, model)
        
        print("Calculating error using loss: ", summary.loss)
        
        func = getattr(keras.losses, summary.loss)
        losses = keras.backend.eval(func(input_data.data, recon))
        
        return losses.tolist()
        
    def get_aucs(self, summary, AUCs_path, filename, data_processor, data_loader, use_qcd_weights_for_signal=False):
        
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
                               use_qcd_weights_for_signal=use_qcd_weights_for_signal)
        
        print("aucs: ", aucs)
        
        append = False
        write_header = True
        
        return (aucs, auc_path, append, write_header)

    def draw_roc_curves(self, summary, data_processor, data_loader, ax, colors, **kwargs):
        model = self.__load_model(summary)

        test_key = "qcd"
        normed = self.__get_normalized_data(summary, data_processor, data_loader, test_key)
        errors = self.__get_losses(normed, model, summary.loss)
        
        signals_errors = {k: v for k, v in errors.items() if k != test_key}
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
         
    def __get_data(self, summary, data_processor, data_loader,
                   test_key="qcd", use_qcd_weights_for_signal=False):
        data = {
            test_key: self.get_qcd_data(summary, data_processor, data_loader, test_data_only=True)
        }
        
        for name, path in self.signal_dict.items():
            if name == test_key: continue
        
            data[name] = self.get_signal_data(path=path,
                                              summary=summary,
                                              data_processor=data_processor,
                                              data_loader=data_loader,
                                              test_data_only=False,
                                              use_qcd_weights=use_qcd_weights_for_signal)

        return data
        
    def __get_losses(self, data, model, summary, test_key):
        losses = {}
    
        for key, normed_data in data.items():
            losses[key] = self.get_error(normed_data, summary, scaler=data[test_key].scaler, model=model)
        return losses
        
    def __get_aucs(self, summary, data_processor, data_loader, model,
                   test_key='qcd', use_qcd_weights_for_signal=False):
    
        self.__update_signals_efp_base(summary, test_key)
        
        normed = self.__get_data(summary, data_processor, data_loader, test_key, use_qcd_weights_for_signal)
        errors = self.__get_losses(normed, model, summary, test_key)

        background_errors = errors[test_key]
        background_labels = [0] * len(background_errors)
        background_weights = normed[test_key].weights
    
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