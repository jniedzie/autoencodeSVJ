import module.SummaryProcessor as summaryProcessor
import module.utils as utils
from module.DataProcessor import DataProcessor
from module.DataHolder import DataHolder
from module.DataLoader import DataLoader
import glob, os
from pathlib import Path
from enum import Enum
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from module.PklFile import PklFile
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json
import keras


class EvaluatorBase:
    
    class ModelTypes(Enum):
        AutoEncoder = 0
        Bdt = 1
    
    def __init__(self, model_type):
        self.model_type = model_type

    def get_data(self, data_path, is_background, data_processor,
                 include_hlf, include_eflow, hlf_to_drop):
    
        data_loader = DataLoader()
    
        (data, _, _, _) = data_loader.load_all_data(data_path, "",
                                                    include_hlf=include_hlf,
                                                    include_eflow=include_eflow,
                                                    hlf_to_drop=hlf_to_drop)
    
        (_, _, data_X_test, _, _) = data_processor.split_to_train_validate_test(data_table=data)
    
        fun = np.zeros if is_background else np.ones
        data_Y_test = pd.DataFrame(fun((len(data_X_test.df), 1)), index=data_X_test.index, columns=['tag'])
    
        return data_X_test, data_Y_test

    def load_model(self, results_file_path):
        config = PklFile(results_file_path+".pkl")
        model = model_from_json(config['model_json'])
        model.load_weights(results_file_path+"_weights.h5")
        return model

    def get_test_dataset(self, data_holder, data_processor, test_key='qcd'):
        qcd = getattr(data_holder, test_key).data
        _, _, test, _, _ = data_processor.split_to_train_validate_test(data_table=qcd)
        return test

    def get_aucs(self, data_holder, data_processor, model, loss_function, norm_type, norm_args = None, test_key='qcd'):
        normed = {}
    
        test = self.get_test_dataset(data_holder, data_processor, test_key)
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

    def save_aucs_autoencoder(self, summary, AUCs_path, filename, data_processor, input_path):
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

        model = self.load_model(summary.training_output_path)
        
        aucs = self.get_aucs(data_holder=data_holder,
                             data_processor=data_processor,
                             model=model,
                             norm_type=summary.norm_type,
                             norm_args=summary.norm_args,
                             loss_function=summary.loss
                             )
        self.save_aucs_to_csv(aucs, auc_path)
    
    def fill_aucs_bdt(self, summary, AUCs_path, filename, data_processor, signals_base_path):
        # TODO: this should skip versions for which all auc were already stored
        # more difficult here because we are storing multiple results in the same file
        
        auc_filename = "_".join(filename.split("_")[0:-3]) + "_" + filename.split("_")[-1]
        auc_path = AUCs_path + "/" + auc_filename
        print("Saving AUC's to path:", auc_path)
        
        qcd_X_test, qcd_Y_test = self.get_data(data_path=summary.qcd_path,
                                               is_background=True,
                                               data_processor=data_processor,
                                               include_hlf=summary.hlf,
                                               include_eflow=summary.eflow,
                                               hlf_to_drop=summary.hlf_to_drop
                                               )
    
        signal_components = filename.split("_")
        mass_index = [i for i, s in enumerate(signal_components) if 'GeV' in s][0]
    
        mass = int(signal_components[mass_index].strip("GeV"))
        rinv = float(signal_components[mass_index + 1])
    
        signal_name = "{}GeV_{:3.2f}".format(mass, rinv)
        signal_path = signals_base_path + "/" + signal_name + "/base_3/*.h5"
    
        model_path = summary.training_output_path + ".weigths"
        try:
            print("Reading file: ", model_path)
            model = open(model_path, 'rb')
        except IOError:
            print("Couldn't open file ", model_path, ". Skipping...")
            return
        bdt = pickle.load(model)
    
        svj_X_test, svj_Y_test = self.get_data(data_path=signal_path,
                                               is_background=False,
                                               data_processor=data_processor,
                                               include_hlf=summary.hlf,
                                               include_eflow=summary.eflow,
                                               hlf_to_drop=summary.hlf_to_drop
                                               )

        X_test = qcd_X_test.append(svj_X_test)
        Y_test = qcd_Y_test.append(svj_Y_test)

        model_auc = roc_auc_score(Y_test, bdt.decision_function(X_test))
        
        print("Area under ROC curve: %.4f" % (model_auc))

        write_header = False if os.path.exists(auc_path) else True

        self.save_aucs_to_csv(aucs=[{"mass": mass, "rinv": rinv, "auc": model_auc}],
                              path=auc_path, append=True, write_header=write_header)
    
    def save_aucs_to_csv(self, aucs, path, append=False, write_header=True):
        # TODO: we could simplify what we store in the aucs file to m, r and auc only
        
        with open(path, "a" if append else "w") as out_file:
            if write_header:
                out_file.write(",name,auc,mass,nu\n")
            
            for index, dict in enumerate(aucs):
                out_file.write("{},Zprime_{}GeV_{},{},{},{}\n".format(index,
                                                                      dict["mass"], dict["rinv"], dict["auc"],
                                                                      dict["mass"], dict["rinv"]))
    
    def save_aucs(self, signals_base_path, AUCs_path, summary_path, input_path):
    
        summaries = summaryProcessor.summary(summary_path=summary_path)

        if not os.path.exists(AUCs_path):
            Path(AUCs_path).mkdir(parents=True, exist_ok=False)

        for index, summary in summaries.df.iterrows():
            utils.set_random_seed(summary.seed)
            filename = summary.training_output_path.split("/")[-1]
    
            data_processor = DataProcessor(validation_fraction=summary.val_split,
                                           test_fraction=summary.test_split,
                                           seed=summary.seed
                                           )
            
            if self.model_type is self.ModelTypes.Bdt:
                
                self.fill_aucs_bdt(summary=summary,
                                   AUCs_path=AUCs_path,
                                   filename=filename,
                                   data_processor=data_processor,
                                   signals_base_path=signals_base_path,
                                   )

            elif self.model_type is self.ModelTypes.AutoEncoder:
                
                self.save_aucs_autoencoder(summary=summary,
                                           AUCs_path=AUCs_path,
                                           filename=filename,
                                           data_processor=data_processor,
                                           input_path=input_path
                                           )