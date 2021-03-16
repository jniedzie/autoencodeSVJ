import module.SummaryProcessor as summaryProcessor
from module.EvaluatorAutoEncoder import EvaluatorAutoEncoder
from module.DataProcessor import DataProcessor
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import importlib, argparse
import module.utils as utils

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
    
    signals = {"{}, {}".format(mass, rinv): "{}{}GeV_{:1.2f}/base_3/*.h5".format(config.signals_base_path, mass, rinv)
               for mass in masses
               for rinv in rinvs}
    return signals


def get_evaluator():
    summaries = summaryProcessor.get_summaries_from_path(config.summary_path)
    
    summary = None
    
    for _, s in summaries.df.iterrows():
        version = summaryProcessor.get_version(s.summary_path)
        if version != config.best_model:
            continue
        summary = s

    utils.set_random_seed(summary.seed)
    data_processor = DataProcessor(summary=summary)
    evaluator = EvaluatorAutoEncoder(input_path=config.input_path)
    
    return evaluator, data_processor, summary


def get_losses():
    evaluator, data_processor, summary = get_evaluator()

    qcd_data = evaluator.get_qcd_test_data(summary=summary, data_processor=data_processor)
    loss_qcd = evaluator.get_error(qcd_data, data_processor=data_processor, summary=summary)
    
    loss_signal = []
    signals = get_signals()
    
    for name, path in signals.items():
        
        signal_data = evaluator.get_signal_test_data(name, path, summary)
        signal_loss = evaluator.get_error(signal_data, data_processor=data_processor, summary=summary)
        loss_signal.append(signal_loss)
    
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