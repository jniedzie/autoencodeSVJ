import glob, os
import tensorflow as tf
import keras
from keras.models import model_from_json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pandas as pd
import pickle

import numpy as np


class EvaluatorPca:
    def __init__(self, input_path, custom_objects):
        
        self.custom_objects = custom_objects
        
        self.signal_dict = {}
        for path in glob.glob(input_path):
            key = path.split("/")[-3]
            self.signal_dict[key] = path

    def get_qcd_data(self, summary, data_processor, data_loader, normalize=False, test_data_only=True):
        
        (data, _, _) = data_loader.load_all_data(globstring=summary.qcd_path, name="QCD")
        if test_data_only:
            (_, _, data) = data_processor.split_to_train_validate_test(data)

        if normalize:
            data = data_processor.normalize(data_table=data,
                                            normalization_type=summary.norm_type,
                                            norm_args=summary.norm_args)
        
        return data
    
    def get_signal_data(self, name, path, summary, data_processor, data_loader, normalize=False, scaler=None, test_data_only=True):
        
        (data, _, _) = data_loader.load_all_data(globstring=path, name=name)
        
        if test_data_only:
            (_, _, data) = data_processor.split_to_train_validate_test(data)
        
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

        reconstructed = model.transform(input_data_normed.data)
        reconstructed = pd.DataFrame(reconstructed)
        reconstructed.index = input_data_normed.index

        return reconstructed

    def __get_mahalanobis_distance(self, inv_cov_matrix, mean_distr, data, verbose=False):
        inv_covariance_matrix = inv_cov_matrix
        vars_mean = mean_distr
        diff = data - vars_mean
        md = []
        for i in range(len(diff)):
            md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
        return md

    def get_error(self, input_data, summary, data_processor, scaler):
        recon = self.get_reconstruction(input_data, summary, data_processor, scaler)

        data_test = np.array(recon.values)
        dist_test = self.__get_mahalanobis_distance(summary.inv_cov_matrix, summary.mean_distribution, data_test, verbose=False)

        return dist_test

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
                               model=model
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
            recon = pd.DataFrame(model.transform(data.data), columns=data.columns, index=data.index, dtype="float64")
            data_test = np.array(recon.values)
            dist_test = self.__get_mahalanobis_distance(summary.inv_cov_matrix, summary.mean_distribution, data_test, verbose=False)
            errors[key] = dist_test

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
                except :
                    print("Failed reading model with model_from_json")
                    try:
                        print("Trying to read model directly from pkl")
                        model = pickle.load(open(model_path, 'rb'))
                    except:
                        print("Failed reading model directly from pkl")
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

    def __get_aucs(self, summary, data_processor, data_loader, model, test_key='qcd'):
        
        normed = {
            test_key: self.get_qcd_data(summary, data_processor, data_loader, normalize=True, test_data_only=True)
        }
    
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
            
            normed[name] = self.get_signal_data(name=name, path=new_path,
                                                summary=summary,
                                                data_processor=data_processor, data_loader= data_loader,
                                                normalize=True, scaler=normed[test_key].scaler)

        errors = {}
    
        for key, data in normed.items():
            model = self.__load_model(summary)

            reconstructed = model.transform(data.data)
            reconstructed = pd.DataFrame(reconstructed)
            reconstructed.index = data.index

            data_test = np.array(reconstructed.values)
            dist_test = self.__get_mahalanobis_distance(summary.inv_cov_matrix, summary.mean_distribution, data_test, verbose=False)
            errors[key] = dist_test
    
        background_errors = errors[test_key]
        background_labels = [0] * len(background_errors)
    
        aucs = []
        for name, signal_err in errors.items():
            if name == test_key: continue
        
            pred = signal_err + background_errors
            true = [1] * len(signal_err)
            true = true + background_labels
        
            auc = roc_auc_score(y_true=true, y_score=pred)
        
            signal_components = name.split("_")
            mass_index = [i for i, s in enumerate(signal_components) if 'GeV' in s][0]
            mass = int(signal_components[mass_index].strip("GeV"))
            rinv = float(signal_components[mass_index + 1])
        
            aucs.append({"mass": mass, "rinv": rinv, "auc": auc})
    
        return aucs