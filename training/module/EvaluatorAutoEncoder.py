import glob, os
import tensorflow as tf
from module.DataHolder import DataHolder
from module.PklFile import PklFile
import keras
from keras.models import model_from_json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pandas as pd
from module.DataLoader import DataLoader
from module.DataTable import DataTable


class EvaluatorAutoEncoder:
    def __init__(self, input_path):
        self.signal_dict = {}
        for path in glob.glob(input_path):
            key = path.split("/")[-3]
            self.signal_dict[key] = path
        
    def get_qcd_test_data(self, summary, data_processor):
        data_holder = DataHolder(qcd=summary.qcd_path, **self.signal_dict)
        data_holder.load()
        return self.__get_test_dataset(data_holder, data_processor)
        
    def get_reconstruction(self, input_data, summary, data_processor):
    
        input_data_normed = data_processor.normalize(data_table=input_data,
                                              normalization_type=summary.norm_type,
                                              norm_args=summary.norm_args)

        model = self.__load_model(summary.training_output_path)
        
        reconstructed = DataTable(pd.DataFrame(model.predict(input_data_normed.data),
                                       columns=input_data_normed.columns,
                                       index=input_data_normed.index,
                                       dtype="float64"))

        reconstructed_denormed = data_processor.normalize(data_table=reconstructed,
                                         normalization_type=summary.norm_type,
                                         norm_args=summary.norm_args,
                                         scaler=input_data.scaler,
                                         inverse=True)
        
        return reconstructed_denormed
        
    def get_aucs(self, summary, AUCs_path, filename, data_processor, **kwargs):
        
        auc_path = AUCs_path + "/" + filename
    
        if os.path.exists(auc_path):
            print("File :", auc_path, "\talready exists. Skipping...")
            return None
        else:
            print("Preparing: ", auc_path)
    
        tf.compat.v1.reset_default_graph()
    
        data_holder = DataHolder(qcd=summary.qcd_path, **self.signal_dict)
        data_holder.load()
    
        model = self.__load_model(summary.training_output_path)
    
        aucs = self.__get_aucs(data_holder=data_holder,
                               data_processor=data_processor,
                               model=model,
                               norm_type=summary.norm_type,
                               norm_args=summary.norm_args,
                               loss_function=summary.loss
                               )
        
        append = False
        write_header = True
        
        return (aucs, auc_path, append, write_header)

    def draw_roc_curves(self, summary, data_processor, ax, colors, signals, **kwargs):
    
        data_holder = DataHolder(qcd=summary.qcd_path, **self.signal_dict)
        data_holder.load()

        qcd_key = "qcd"
        model = self.__load_model(summary.training_output_path)
        
        qcd_data = self.__get_test_dataset(data_holder, data_processor, qcd_key)
        qcd_normed = data_processor.normalize(data_table=qcd_data, normalization_type=summary.norm_type, norm_args=summary.norm_args)

        data_loader = DataLoader()
        
        all_data = {qcd_key: qcd_normed}

        for signal in signals:
            
            (data, _, _, _) = data_loader.load_all_data(signals[signal],
                                                        signal,
                                                        include_hlf=summary.hlf,
                                                        include_eflow=summary.eflow,
                                                        hlf_to_drop=summary.hlf_to_drop,
                                                        )
            
            data_norm = data_processor.normalize(data_table=data,
                                                 normalization_type=summary.norm_type,
                                                 norm_args=summary.norm_args,
                                                 scaler=qcd_data.scaler)
            
            # put normalized signal in "data"
            all_data[signal] = data_norm

        errors = {}

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
        
    def __load_model(self, results_file_path):
        config = PklFile(results_file_path + ".pkl")
        model = model_from_json(config['model_json'])
        model.load_weights(results_file_path + "_weights.h5")
        return model

    def __get_test_dataset(self, data_holder, data_processor, test_key='qcd'):
        qcd = getattr(data_holder, test_key).data
        _, _, test = data_processor.split_to_train_validate_test(data_table=qcd)
        return test

    def __get_aucs(self, data_holder, data_processor, model, loss_function, norm_type, norm_args=None, test_key='qcd'):
        normed = {}
    
        test = self.__get_test_dataset(data_holder, data_processor, test_key)
        normed[test_key] = data_processor.normalize(data_table=test, normalization_type=norm_type, norm_args=norm_args)
        qcd_scaler = test.scaler
    
        for key in data_holder.KEYS:
            if key == test_key: continue
            data = getattr(data_holder, key).data
            normed[key] = data_processor.normalize(data_table=data, normalization_type=norm_type, norm_args=norm_args,
                                                   scaler=qcd_scaler)
    
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