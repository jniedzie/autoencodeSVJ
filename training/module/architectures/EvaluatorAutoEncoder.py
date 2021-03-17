import glob, os
import tensorflow as tf
import keras
from keras.models import model_from_json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pandas as pd
import pickle


class EvaluatorAutoEncoder:
    def __init__(self, input_path):
        self.signal_dict = {}
        for path in glob.glob(input_path):
            key = path.split("/")[-3]
            self.signal_dict[key] = path
        
    def get_qcd_test_data(self, summary, data_processor, data_loader, normalize=False):
        
        (data, _, _, _) = data_loader.load_all_data(globstring=summary.qcd_path, name="QCD")
        (_, _, test) = data_processor.split_to_train_validate_test(data)

        if normalize:
            test = data_processor.normalize(data_table=test,
                                            normalization_type=summary.norm_type,
                                            norm_args=summary.norm_args)
        
        return test
    
    def get_signal_test_data(self, name, path, summary, data_processor, data_loader, normalize=False, scaler=None):
        
        (data, _, _, _) = data_loader.load_all_data(globstring=path, name=name)
        (_, _, test) = data_processor.split_to_train_validate_test(data)
        
        if normalize:
            test = data_processor.normalize(data_table=test,
                                            normalization_type=summary.norm_type,
                                            norm_args=summary.norm_args,
                                            scaler=scaler)
        
        return test
        
    def get_reconstruction(self, input_data, summary, data_processor):
    
        input_data_normed = data_processor.normalize(data_table=input_data,
                                                     normalization_type=summary.norm_type,
                                                     norm_args=summary.norm_args)

        model = self.__load_model(summary)
        
        reconstructed = pd.DataFrame(model.predict(input_data_normed.data),
                                     columns=input_data_normed.columns,
                                     index=input_data_normed.index,
                                     dtype="float64")

        reconstructed_denormed = data_processor.normalize(data_table=reconstructed,
                                                          normalization_type=summary.norm_type,
                                                          norm_args=summary.norm_args,
                                                          scaler=input_data.scaler,
                                                          inverse=True)
        
        return reconstructed_denormed
        
    def get_error(self, input_data, summary, data_processor):
        recon = self.get_reconstruction(input_data, summary, data_processor)
        
        func = getattr(keras.losses, summary.loss)
        losses = keras.backend.eval(func(input_data.data, recon))
        error = keras.backend.eval(losses.tolist())
        
        return error
        
    def get_aucs(self, summary, AUCs_path, filename, data_processor, data_loader, **kwargs):
        
        auc_path = AUCs_path + "/" + filename
    
        if os.path.exists(auc_path):
            print("File :", auc_path, "\talready exists. Skipping...")
            return None
        else:
            print("Preparing: ", auc_path)
    
        tf.compat.v1.reset_default_graph()
    
        model = self.__load_model(summary)
    
        aucs = self.__get_aucs(summary=summary,
                               data_processor=data_processor,
                               data_loader=data_loader,
                               model=model,
                               loss_function=summary.loss
                               )
        
        append = False
        write_header = True
        
        return (aucs, auc_path, append, write_header)

    def draw_roc_curves(self, summary, data_processor, data_loader, ax, colors, signals, **kwargs):

        qcd_key = "qcd"
        all_data = {
            qcd_key: self.get_qcd_test_data(summary, data_processor, data_loader, normalize=True)
        }

        for name, path in signals.items():
            all_data[name] = self.get_signal_test_data(name, path, summary, data_processor, data_loader,
                                                       normalize=True, scaler = all_data[qcd_key].scaler)
        
        errors = {}
        model = self.__load_model(summary)

        for key, data in all_data.items():
            recon = pd.DataFrame(model.predict(data.data), columns=data.columns, index=data.index, dtype="float64")
            func = getattr(keras.losses, summary.loss)
            losses = keras.backend.eval(func(data.data, recon))
            errors[key] = keras.backend.eval(losses.tolist())

        signals_errors = {}
        
        for signal in signals:
            signals_errors[signal] = errors[signal]

        qcd_errors = errors[qcd_key]

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
    
        model_path = summary.training_output_path + ".pkl"
        
        try:
            print("Reading file: ", model_path)
            model_file = open(model_path, 'rb')
        except IOError:
            print("Couldn't open file ", model_path, ". Skipping...")
            return None
    
        model = model_from_json(pickle.load(model_file))
        model.load_weights(summary.training_output_path + "_weights.h5")
        
        return model

    def __get_aucs(self, summary, data_processor, data_loader, model, loss_function, test_key='qcd'):
        
        normed = {
            test_key: self.get_qcd_test_data(summary, data_processor, data_loader, normalize=True)
        }
    
        for name, path in self.signal_dict.items():
            if name == test_key: continue

            normed[name] = self.get_signal_test_data(name=name, path=path,
                                                     summary=summary,
                                                     data_processor=data_processor, data_loader= data_loader,
                                                     normalize=True, scaler=normed[test_key].scaler)

        errors = {}
    
        for key, data in normed.items():
            recon = pd.DataFrame(model.predict(data.data), columns=data.columns, index=data.index, dtype="float64")
            func = getattr(keras.losses, loss_function)
            losses = keras.backend.eval(func(data.data, recon))
            errors[key] = keras.backend.eval(losses.tolist())
    
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