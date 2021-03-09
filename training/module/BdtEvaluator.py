import module.utils as utils
from module.Trainer import Trainer
import module.SummaryProcessor as summaryProcessor
from module.DataProcessor import DataProcessor
from module.DataLoader import DataLoader
from module.AucGetter import AucGetter
from module.DataHolder import DataHolder
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
from collections import OrderedDict as odict
from pathlib import Path
import tensorflow as tf
import glob, os
import pickle

class BdtEvaluator:
    
    def __init__(self):
        pass
    
    def draw_ROCs(self, metrics=None, figsize=8, figloc=(0.3, 0.2), *args, **kwargs):
        
        if metrics is None:
            metrics=['mae', 'mse']
        
        qcd = self.errs_dict['qcd']
        others = [self.errs_dict[n] for n in self.all_names if n != 'qcd']
        
        utils.roc_auc_plot(qcd, others, metrics=metrics, figsize=figsize, figloc=figloc, *args, **kwargs)
        return

    @staticmethod
    def get_data(data_path, is_background, data_processor,
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

    @staticmethod
    def save_AUCs(signals_base_path, AUCs_path, summary_path):
        
        summaries = summaryProcessor.summary(summary_path=summary_path)
        
        if not os.path.exists(AUCs_path):
            Path(AUCs_path).mkdir(parents=True, exist_ok=False)

        for index, row in summaries.df.iterrows():
            utils.set_random_seed(row.seed)

            path = row.training_output_path
            model_path = path + ".weigths"
            filename = path.split("/")[-1]
            auc_path = AUCs_path + "/" + filename
            print(auc_path)

            if os.path.exists(auc_path):
                print("File :", auc_path, "\talready exists. Skipping...")
                continue

            data_processor = DataProcessor(validation_fraction=row.val_split,
                                           test_fraction=row.test_split,
                                           seed=row.seed
                                           )

            qcd_X_test, qcd_Y_test = BdtEvaluator.get_data(data_path=row.qcd_path,
                                                           is_background=True,
                                                           data_processor=data_processor,
                                                           include_hlf=row.hlf,
                                                           include_eflow=row.eflow,
                                                           hlf_to_drop=row.hlf_to_drop
                                                           )

            signal_index = 0

            with open(auc_path, "w") as out_file:
                out_file.write(",name,auc,mass,nu\n")
                
                base_filename = auc_path.split("/")[-1]
                signal_components = base_filename.split("_")
                mass = int(signal_components[-3].strip("GeV"))
                rinv = float(signal_components[-2])
            
                signal_name = "{}GeV_{:3.2f}".format(mass, rinv)
                signal_path = signals_base_path + "/" + signal_name + "/base_3/*.h5"
                
                try:
                    print("Reading file: ", model_path)
                    model = open(model_path, 'rb')
                except IOError:
                    print("Couldn't open file ", model_path, ". Skipping...")
                    continue
                bdt = pickle.load(model)
            
                svj_X_test, svj_Y_test = BdtEvaluator.get_data(data_path=signal_path,
                                                               is_background=False,
                                                               data_processor=data_processor,
                                                               include_hlf=row.hlf,
                                                               include_eflow=row.eflow,
                                                               hlf_to_drop=row.hlf_to_drop
                                                               )
            
                X_test = qcd_X_test.append(svj_X_test)
                Y_test = qcd_Y_test.append(svj_Y_test)
            
                model_auc = roc_auc_score(Y_test, bdt.decision_function(X_test))
                print("Area under ROC curve: %.4f" % (model_auc))
            
                out_file.write(
                    "{},Zprime_{}GeV_{},{},{},{}\n".format(signal_index, mass, rinv, model_auc, mass, rinv))
                signal_index += 1
        
        

        # masses = [1500, 2000, 2500, 3000, 3500, 4000]
        # rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
        
        # signal_index=0
        #
        # AUC_file_path = AUCs_path + "/" + self.file_name + "_v{}".format(version)
        #
        # if os.path.exists(AUC_file_path):
        #     print("File :", AUC_file_path, "\talready exists. Skipping...")
        #     return
        #
        # if not os.path.exists(AUCs_path):
        #     Path(AUCs_path).mkdir(parents=True, exist_ok=False)
        #
        # with open(AUC_file_path, "w") as out_file:
        #     out_file.write(",name,auc,mass,nu\n")
        #     for mass in masses:
        #         for rinv in rinvs:
        #             signal_name = "{}GeV_{:3.2f}".format(mass, rinv)
        #             signal_path = signals_base_path + "/" + signal_name + "/base_3/*.h5"
        #             base_filename = self.file_name + "_{}_v{}".format(signal_name, version)
        #             training_base_path = results_path + base_filename
        #
        #             print("Reading file: ", training_base_path, ".weigths")
        #
        #             try:
        #                 model = open(training_base_path + ".weigths", 'rb')
        #             except IOError:
        #                 print("Couldn't open file ", training_base_path, ".weigths. Skipping...")
        #                 continue
        #             bdt = pickle.load(model)
        #
        #             svj_X_test, svj_Y_test = self.get_data(data_path=signal_path, is_background=False)
        #
        #             X_test = qcd_X_test.append(svj_X_test)
        #             Y_test = qcd_Y_test.append(svj_Y_test)
        #
        #
        #             model_auc = roc_auc_score(Y_test, bdt.decision_function(X_test))
        #             print("Area under ROC curve: %.4f" % (model_auc))
        #
        #             out_file.write("{},Zprime_{}GeV_{},{},{},{}\n".format(signal_index, mass, rinv, model_auc, mass, rinv))
        #             signal_index += 1