import module.SummaryProcessor as summaryProcessor
import module.utils as utils
import module.AutoEncoderTrainerHelper as trainer
from module.DataProcessor import DataProcessor
from module.DataHolder import DataHolder

from pathlib import Path
import tensorflow as tf
import glob, os

import pandas as pd
import numpy as np


class AucGetter(object):
    """This object basically needs to be able to load a training run into memory, including all
    training/testing fractions and random seeds. It then should take a library of signals as input
    and be able to evaluate the auc on each signal to determine a 'general auc' for all signals.
    """
    
    def __init__(self, filename, summary_path):
        self.set_variables_from_summary(summary_path+filename)
        
        if not os.path.exists(self.training_output_path + ".pkl"):
            print("ERROR -- AucGetter requires .pkl file, but it was not found:", self.training_output_path, ".pkl")
        
        self.trainer = trainer.AutoEncoderTrainerHelper(self.training_output_path)
    
    def set_variables_from_summary(self, path):
        name = summaryProcessor.summary_by_name(path)
        summary_data = summaryProcessor.load_summary(name)
        
        self.hlf = summary_data['hlf']
        self.eflow = summary_data['eflow']
        self.eflow_base = summary_data['eflow_base']
        self.hlf_to_drop = list(map(str, summary_data['hlf_to_drop']))
        self.seed = summary_data['seed']
        self.test_split = summary_data['test_split']
        self.validation_split = summary_data['val_split']
        self.qcd_path = summary_data['qcd_path']
        self.training_output_path = summary_data['training_output_path']
        self.norm_type = summary_data["norm_type"]
        self.norm_ranges = np.asarray(summary_data["range"]) if "range" in summary_data.keys() else None
        self.norm_args = summary_data['norm_args']
        
    def get_test_dataset(self, data_holder, test_key='qcd'):
        
        utils.set_random_seed(self.seed)
        
        qcd = getattr(data_holder, test_key).data
        
        data_processor = DataProcessor(validation_fraction=self.validation_split,
                                       test_fraction=self.test_split,
                                       seed=self.seed)
        
        _, _, test, _, _ = data_processor.split_to_train_validate_test(data_table=qcd)
        
        return test
    
    def get_errs_recon(self, data_holder, test_key='qcd', **kwargs):
        
        data_processor = DataProcessor(seed=self.seed)
        normed = {}

        
        means = {}
        stds = {}

        test = self.get_test_dataset(data_holder, test_key)
        
        if self.norm_type == "CustomStandard":
            means[test_key], stds[test_key] = test.get_means_and_stds()
        else:
            means[test_key], stds[test_key] = None, None

        
        normed[test_key]= data_processor.normalize(data_table=test,
                                                   normalization_type=self.norm_type,
                                                   data_ranges=self.norm_ranges,
                                                   norm_args=self.norm_args,
                                                   means=means[test_key],
                                                   stds=stds[test_key])


        qcd_scaler = test.scaler

        for key in data_holder.KEYS:
            if key != test_key:
                data = getattr(data_holder, key).data

                if self.norm_type == "CustomStandard":
                    means[key], stds[key] = data.get_means_and_stds()
                else:
                    means[key], stds[key] = None, None
                
                normed[key] = data_processor.normalize(data_table=data,
                                                       normalization_type=self.norm_type,
                                                       data_ranges=self.norm_ranges,
                                                       norm_args=self.norm_args,
                                                       means=means[key],
                                                       stds=stds[key],
                                                       scaler=qcd_scaler)
        
        for key in normed:
            normed[key].name = key
        
        model = self.trainer.load_model()
        
        err, recon = utils.get_recon_errors(normed, model, **kwargs)
        
        for key, value in err.items():
            err[key].name = value.name.rstrip('error').strip()
        
        for key, value in recon.items():
            recon[key] = data_processor.normalize(data_table=value,
                                                  normalization_type=self.norm_type,data_ranges=self.norm_ranges,
                                                  norm_args=self.norm_args,
                                                  inverse=True,
                                                  means=means[key],
                                                  stds=stds[key],
                                                  scaler=qcd_scaler)

        del model
        
        return [{z.name: z for y, z in x.items()} for x in [normed, err, recon]]
    
    def get_aucs(self, errors, qcd_key='qcd', metrics=None):
        
        if metrics is None:
            metrics = ['mae']

        background_errors = []
        signal_errors = []

        for key, value in list(errors.items()):
            if key == qcd_key:
                background_errors.append(value)
            else:
                signal_errors.append(value)

        ROCs_and_AUCs_per_signal = utils.roc_auc_dict(data_errs=background_errors, signal_errs=signal_errors, metrics=metrics)
        
        return ROCs_and_AUCs_per_signal
    
    def auc_metric(self, aucs):
        data = [(k, v['mae']['auc']) for k, v in list(aucs.items())]
        fmt = pd.DataFrame(data, columns=['name', 'auc'])

        new_list = []
        
        for x in fmt.name:
            mass_and_r = []
            for y in x.split('_')[1:]:
                variable = y.rstrip('GeV')
                variable = variable.replace("p", ".")
                
                mass_and_r.append(float(variable))
            
            new_list.append(mass_and_r)
        
        mass, nu = np.asarray(new_list).T
        nu /= 100
        
        fmt['mass'] = mass
        fmt['nu'] = nu
        
        return fmt

    @staticmethod
    def save_AUCs(input_path, AUCs_path, summary_path):
    
        signal_dict = {}
        for path in glob.glob(input_path):
            key = path.split("/")[-3]
            signal_dict[key] = path
    
        summaries = summaryProcessor.summary(summary_path=summary_path)
    
        if not os.path.exists(AUCs_path):
            Path(AUCs_path).mkdir(parents=True, exist_ok=False)
    
        for index, row in summaries.df.iterrows():
            path = row.training_output_path
            filename = path.split("/")[-1]
            auc_path = AUCs_path + "/" + filename
        
            if not os.path.exists(auc_path):
                tf.compat.v1.reset_default_graph()
            
                auc_getter = AucGetter(filename=filename, summary_path=summary_path)
            
                data_holder = DataHolder(qcd=row.qcd_path, **signal_dict)
                data_holder.load()
            
                norm, err, recon = auc_getter.get_errs_recon(data_holder)
            
                ROCs = auc_getter.get_aucs(err)
                AUCs = auc_getter.auc_metric(ROCs)
                AUCs.to_csv(auc_path)