from module.DataLoader import DataLoader
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
import os
import numpy as np
import pandas as pd


class EvaluatorBdt:
    def __init__(self, **kwargs):
        pass

    def get_aucs(self, summary, AUCs_path, filename, data_processor, signals_base_path):
        # TODO: this should skip versions for which all auc were already stored
        # more difficult here because we are storing multiple results in the same file
    
        auc_filename = "_".join(filename.split("_")[0:-3]) + "_" + filename.split("_")[-1]
        auc_path = AUCs_path + "/" + auc_filename
        print("Saving AUC's to path:", auc_path)
    
        mass, rinv = self.__get_mass_and_rinv_from_filename(filename)
    
        signal_name = "{}GeV_{:3.2f}".format(mass, rinv)
        signal_path = signals_base_path + "/" + signal_name + "/base_3/*.h5"
    
        model = self.__get_model(summary)
        if model is None:
            return None
        
        self.__load_data(data_processor=data_processor, summary=summary, signal_path=signal_path)
    
        model_auc = roc_auc_score(self.Y_test, model.decision_function(self.X_test))
    
        print("Area under ROC curve: %.4f" % (model_auc))

        aucs = [{"mass": mass, "rinv": rinv, "auc": model_auc}]
        append = True
        write_header = False if os.path.exists(auc_path) else True
    
        return (aucs, auc_path, append, write_header)

    def draw_roc_curves(self, summary, filename, data_processor, signals_base_path, ax, colors):
    
        mass, rinv = self.__get_mass_and_rinv_from_filename(filename)
    
        signal_name = "{}GeV_{:3.2f}".format(mass, rinv)
        signal_path = signals_base_path + "/" + signal_name + "/base_3/*.h5"
    
        self.__load_data(data_processor=data_processor, summary=summary, signal_path=signal_path)
    
        model = self.__get_model(summary)
        if model is None:
            return None
    
        auc = roc_auc_score(y_true=self.Y_test, y_score=model.decision_function(self.X_test))
        roc = roc_curve(y_true=self.Y_test, y_score=model.decision_function(self.X_test))
    
        i = 0
        ax.plot(roc[0], roc[1], "-", c=colors[i % len(colors)], label='{}, AUC {:.4f}'.format(filename, auc))

    def __load_data(self, data_processor, summary, signal_path):
        qcd_X_test, qcd_Y_test = self.__get_data(data_path=summary.qcd_path,
                                                 is_background=True,
                                                 data_processor=data_processor,
                                                 summary=summary
                                                 )

        svj_X_test, svj_Y_test = self.__get_data(data_path=signal_path,
                                                 is_background=False,
                                                 data_processor=data_processor,
                                                 summary=summary
                                                 )

        self.X_test = qcd_X_test.append(svj_X_test)
        self.Y_test = qcd_Y_test.append(svj_Y_test)
        
    def __get_mass_and_rinv_from_filename(self, filename):
        signal_components = filename.split("_")
        mass_index = [i for i, s in enumerate(signal_components) if 'GeV' in s][0]
    
        mass = int(signal_components[mass_index].strip("GeV"))
        rinv = float(signal_components[mass_index + 1])
        
        return mass, rinv

    def __get_model(self, summary):
        model_path = summary.training_output_path + ".weigths"
        try:
            print("Reading file: ", model_path)
            model = open(model_path, 'rb')
        except IOError:
            print("Couldn't open file ", model_path, ". Skipping...")
            return None
        
        return pickle.load(model)

    def __get_data(self, data_path, is_background, data_processor, summary):
    
        data_loader = DataLoader()
    
        (data, _, _, _) = data_loader.load_all_data(data_path, "",
                                                    include_hlf=summary.hlf,
                                                    include_eflow=summary.eflow,
                                                    hlf_to_drop=summary.hlf_to_drop)
    
        (_, _, data_X_test) = data_processor.split_to_train_validate_test(data_table=data)
    
        fun = np.zeros if is_background else np.ones
        data_Y_test = pd.DataFrame(fun((len(data_X_test.df), 1)), index=data_X_test.index, columns=['tag'])
    
        return data_X_test, data_Y_test
