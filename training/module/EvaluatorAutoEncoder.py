import glob, os
import tensorflow as tf
from module.DataHolder import DataHolder
import module.utils as utils
from module.PklFile import PklFile
import keras
from keras.models import model_from_json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
from collections import OrderedDict as odict
from module.DataLoader import DataLoader
import matplotlib.pyplot as plt

class EvaluatorAutoEncoder:
    def __init__(self, input_path):
        self.signal_dict = {}
        for path in glob.glob(input_path):
            key = path.split("/")[-3]
            self.signal_dict[key] = path
        
    def get_qcd_test_data(self):
        pass
        

    def save_aucs(self, summary, AUCs_path, filename, data_processor):
        
        auc_path = AUCs_path + "/" + filename
    
        if os.path.exists(auc_path):
            print("File :", auc_path, "\talready exists. Skipping...")
            return
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
        utils.save_aucs_to_csv(aucs, auc_path)

    def draw_roc_curves(self, summary, data_processor, signals, figsize=8, figloc=(0.3, 0.2), *args, **kwargs):
    
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

        self.__roc_auc_plot(qcd_errors, signals_errors, figsize=figsize, figloc=figloc, *args, **kwargs)

    def __roc_auc_plot(self, background_errors, signal_errs, *args, **kwargs):
    
        background_labels = [0] * len(background_errors)
        
        fig, ax_begin, ax_end, plt_end, colors = self.__get_plot_params(1, *args, **kwargs)
        ax = ax_begin(0)
        
        i = 0
        roc = []
        
        for name, signal_err in signal_errs.items():
            pred = signal_err + background_errors
            true = [1] * len(signal_err)
            true = true + background_labels
            
            roc = roc_curve(y_true=true, y_score=pred)
            auc = roc_auc_score(y_true=true, y_score=pred)
            
            ax.plot(roc[0], roc[1], "-", c=colors[i % len(colors)], label='{}, AUC {:.4f}'.format(name, auc))
            
            i += 1
    
        ax.plot(roc[0], roc[0], '--', c='black')
        ax_end("false positive rate", "true positive rate")
        plt_end()
        plt.show()

    def __get_plot_params(self,
            n_plots,
            cols=4,
            figsize=20.,
            yscale='linear',
            xscale='linear',
            figloc='lower right',
            figname='Untitled',
            savename=None,
            ticksize=8,
            fontsize=5,
            colors=None
    ):
        rows = n_plots / cols + bool(n_plots % cols)
        if n_plots < cols:
            cols = n_plots
            rows = 1
    
        if not isinstance(figsize, tuple):
            figsize = (figsize, rows * float(figsize) / cols)
    
        fig = plt.figure(figsize=figsize)
    
        def on_axis_begin(i):
            return plt.subplot(rows, cols, i + 1)
    
        def on_axis_end(xname, yname=''):
            plt.xlabel(xname + " ({0}-scaled)".format(xscale))
            plt.ylabel(yname + " ({0}-scaled)".format(yscale))
            plt.xticks(size=ticksize)
            plt.yticks(size=ticksize)
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.gca().spines['left']._adjust_location()
            plt.gca().spines['bottom']._adjust_location()
    
        def on_plot_end():
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = odict(list(zip(list(map(str, labels)), handles)))
            plt.figlegend(list(by_label.values()), list(by_label.keys()), loc=figloc)
            # plt.figlegend(handles, labels, loc=figloc)
            plt.suptitle(figname)
            plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01, rect=[0, 0.03, 1, 0.95])
            if savename is None:
                plt.show()
            else:
                plt.savefig(savename)
    
        if colors is None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if len(colors) < n_plots:
            print("too many plots for specified colors. overriding with RAINbow")
            import matplotlib.cm as cm
            colors = cm.rainbow(np.linspace(0, 1, n_plots))
        return fig, on_axis_begin, on_axis_end, on_plot_end, colors
    
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