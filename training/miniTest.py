import module.utils as utils
import module.evaluate as ev
import module.summaryProcessor as summaryProcessor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime
import os
import glob

output_path = "trainingResults/"
summary_path = output_path+"summary/"
results_path = output_path+"trainingRuns/"

input_path = "../../data/training_data/"
signal_path = input_path + "all_signals/2000GeV_0.15/base_3/*.h5"
qcd_path = input_path + "qcd/base_3/*.h5"

def get_latest_summary_file_path(efp_base, bottleneck_dim, version=None):
    if version is None:
        version = summaryProcessor.get_last_summary_file_version(output_path, "hlf_eflow{}_{}_".format(efp_base, bottleneck_dim))

    input_summary_path = summary_path+"/hlf_eflow{}_{}_v{}.summary".format(efp_base, bottleneck_dim, version)
    return input_summary_path


def draw_ROC_curves(efp_base, bottleneck_dim, version=None):
    input_summary_path = get_latest_summary_file_path(efp_base, bottleneck_dim, version)
    print("Drawing ROC curve for summary: ", input_summary_path)

    # masses = [1500, 2000, 2500, 3000, 3500, 4000]
    masses = [2000]
    rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
    base_path = "../../data/training_data/all_signals/"
    aux_signals = {"{}, {}".format(mass, rinv) : "{}{}GeV_{:1.2f}/base_3/*.h5".format(base_path, mass, rinv) for mass in masses for rinv in rinvs}

    # elt = ev.ae_evaluation(input_summary_path, qcd_path=qcd_path, SVJ_path=signal_path)
    elt = ev.ae_evaluation(input_summary_path, qcd_path=qcd_path, SVJ_path=signal_path, aux_signals_dict=aux_signals)
    elt.roc(xscale='log', metrics=["mae"])


 # ev.ae_train(signal_path=signal_path,
 #             qcd_path=qcd_path,
 #             output_data_path=output_path,
 #             target_dim=8,
 #             verbose=1,
 #             batch_size=32,
 #             learning_rate=0.00051,
 #             epochs=2,
 #             es_patience=12,
 #             lr_patience=9,
 #             lr_factor=0.5,
 #             norm_percentile=25
 #             )


ev.save_all_missing_AUCs(summary_path=summary_path,
                         AUCs_path=output_path + "/aucs",
                         qcd_path=qcd_path,
                         signals_path=(input_path + "all_signals/*/base_3/*.h5"))

# draw_ROC_curves(efp_base=3, bottleneck_dim=8)