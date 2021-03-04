import module.SummaryProcessor as summaryProcessor
from module.AutoEncoderEvaluator import AutoEncoderEvaluator
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import importlib, argparse

# ------------------------------------------------------------------------------------------------
# This script will draw reconstruction loss for a a mixture of all models found in the
# "signals_base_path" and backgrounds as specified in the training summary file.
# ------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("-c", "--config", dest="config_path", default=None, required=True, help="Path to the config file")
args = parser.parse_args()
config_path = args.config_path.strip(".py").replace("/", ".")
config = importlib.import_module(config_path)

# masses = [1500, 2000, 2500, 3000, 3500, 4000]
masses = [2500]
# rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
rinvs = [0.45]

def get_signals():
    signals = {"signal_{}_{}".format(mass, rinv).replace(".", "p"):
                   "{}{}GeV_{:1.2f}/base_3/*.h5".format(config.signals_base_path, mass, rinv)
               for mass in masses
               for rinv in rinvs}
    
    return signals


def get_evaluator():
    
    input_summary_path = summaryProcessor.get_latest_summary_file_path(summaries_path=config.summary_path,
                                                                       file_name_base=config.file_name,
                                                                       version=config.best_model)
    
    signals = get_signals()
    return AutoEncoderEvaluator(input_summary_path, signals=signals)


def get_losses():
    evaluator = get_evaluator()
    
    loss_qcd = evaluator.qcd_err.mae
    loss_signal = []
    
    for signal in get_signals():
        signal_mae_array = getattr(evaluator, "{}_err".format(signal)).mae
        loss_signal.append(signal_mae_array)
    
    loss_signal = pd.concat(loss_signal)
    
    return loss_qcd, loss_signal


n_columns = 2
n_rows = 2

canvas = plt.figure(figsize=(10, 10))
i_plot = 1

loss_hist = canvas.add_subplot(n_rows, n_columns, i_plot)
i_plot += 1

loss_qcd, loss_signal = get_losses()

loss_hist.hist(loss_qcd, bins=numpy.linspace(0, 0.4, 100), label="qcd", histtype="step", density=True)
loss_hist.hist(loss_signal, bins=numpy.linspace(0, 0.4, 100), label="signal", histtype="step", density=True)
loss_hist.set_yscale("log")
loss_hist.set_ylim(bottom=1E-2, top=1E2)
loss_hist.legend(loc='upper right')

plt.show()