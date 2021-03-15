import glob, os
import tensorflow as tf
from module.DataHolder import DataHolder
import module.utils as utils
from module.PklFile import PklFile
import keras
from keras.models import model_from_json
from sklearn.metrics import roc_auc_score
import pandas as pd

class EvaluatorAutoEncoder:
    def __init__(self):
        pass

    def save_aucs(self, summary, AUCs_path, filename, data_processor, input_path):
        signal_dict = {}
        for path in glob.glob(input_path):
            key = path.split("/")[-3]
            signal_dict[key] = path
    
        auc_path = AUCs_path + "/" + filename
    
        if os.path.exists(auc_path):
            print("File :", auc_path, "\talready exists. Skipping...")
            return
        else:
            print("Preparing: ", auc_path)
    
        tf.compat.v1.reset_default_graph()
    
        data_holder = DataHolder(qcd=summary.qcd_path, **signal_dict)
        data_holder.load()
    
        model = self.__load_model(summary.training_output_path)
    
        aucs = self.__get_aucs(data_holder=data_holder,
                             data_processor=data_processor,
                             model=model,
                             norm_type=summary.norm_type,
                             norm_args=summary.norm_args,
                             loss_function=summary.loss
                             )
        utils.save_aucs_to_csv(aucs, auc_path)

    def __load_model(self, results_file_path):
        config = PklFile(results_file_path + ".pkl")
        model = model_from_json(config['model_json'])
        model.load_weights(results_file_path + "_weights.h5")
        return model

    def __get_test_dataset(self, data_holder, data_processor, test_key='qcd'):
        qcd = getattr(data_holder, test_key).data
        _, _, test, _, _ = data_processor.split_to_train_validate_test(data_table=qcd)
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