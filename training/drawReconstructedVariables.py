import module.SummaryProcessor as summaryProcessor
from module.AutoEncoderEvaluator import AutoEncoderEvaluator
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import importlib, argparse

# ------------------------------------------------------------------------------------------------
# This script will draw input and reconstructed variables for signal found in the
# "signals_base_path" and background as specified in the training summary.
# ------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("-c", "--config", dest="config_path", default=None, required=True, help="Path to the config file")
args = parser.parse_args()
config = importlib.import_module(args.config_path)

input_summary_path = summaryProcessor.get_latest_summary_file_path(
    summaries_path=config.summary_path,
    file_name_base=config.file_name,
    version=config.best_model
)
    
print("Loading summary: ", input_summary_path)

# masses = [1500, 2000, 2500, 3000, 3500, 4000]
masses = [2500]
# rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
rinvs = [0.45]

signals = {"signal_{}_{}".format(mass, rinv).replace(".", "p") : "{}{}GeV_{:1.2f}/base_3/*.h5".format(config.signals_base_path, mass, rinv)
           for mass in masses
           for rinv in rinvs}

n_columns = 5
n_rows = 4

canvas = plt.figure()

i_plot = 1

bins = {
    "Eta" : (-3.5, 3.5, 100),
    "Phi" : (-3.5, 3.5, 100),
    "Pt"  : (0, 2000, 100),
    "M"  : (0, 800, 100),
    "ChargedFraction"  : (0, 1, 100),
    "PTD"  : (0, 1, 100),
    "Axis2"  : (0, 0.2, 100),
    # "Flavor"  : (0, 1000, 100),
    # "Energy"  : (0, 1000, 100),
    "eflow 1"  : (0, 1, 100),
    "eflow 2"  : (0, 1, 100),
    "eflow 3"  : (0, 1, 100),
    "eflow 4"  : (0, 1, 100),
    "eflow 5"  : (0, 1, 100),
    "eflow 6"  : (0, 1, 100),
    "eflow 7"  : (0, 1, 100),
    "eflow 8"  : (0, 1, 100),
    "eflow 9"  : (0, 1, 100),
    "eflow 10"  : (0, 1, 100),
    "eflow 11"  : (0, 1, 100),
    "eflow 12"  : (0, 1, 100),
}


def draw_histogram_for_variable(input_data, reconstructed_data, variable_name, i_plot):
    hist = canvas.add_subplot(n_rows, n_columns, i_plot)
    
    input_values = input_data[variable_name]
    reconstructed_values = reconstructed_data[variable_name]
    
    hist.hist(input_values, bins=numpy.linspace(*bins[variable_name]), alpha=0.5, label='input', histtype="step", density=True)
    hist.hist(reconstructed_values, bins=numpy.linspace(*bins[variable_name]), alpha=0.5, label='reconstruction', histtype="step", density=True)
    hist.title.set_text(variable_name)


evaluator = AutoEncoderEvaluator(input_summary_path, signals=signals)

for variable_name in bins:
    draw_histogram_for_variable(input_data=evaluator.qcd_test_data,
                                reconstructed_data=evaluator.qcd_recon,
                                variable_name=variable_name, i_plot=i_plot)
    i_plot += 1


legend = canvas.add_subplot(n_rows, n_columns, i_plot)
legend.hist([], alpha=0.5, label='input', histtype="step")
legend.hist([], alpha=0.5, label='reconstruction', histtype="step")
legend.legend(loc='upper right')

plt.show()