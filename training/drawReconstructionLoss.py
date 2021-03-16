import module.SummaryProcessor as summaryProcessor
from module.Evaluator import Evaluator
import matplotlib.pyplot as plt
import numpy
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
    
    signals = {"{}, {}".format(mass, rinv): "{}{}GeV_{:1.2f}/base_3/*.h5".format(config.signals_base_path, mass, rinv)
               for mass in masses
               for rinv in rinvs}
    return signals


def get_summary():
    summaries = summaryProcessor.get_summaries_from_path(config.summary_path)
    
    summary = None
    
    for _, s in summaries.df.iterrows():
        version = summaryProcessor.get_version(s.summary_path)
        if version != config.best_model:
            continue
        summary = s
        
    return summary


def get_losses():
    evaluator = Evaluator(model_evaluator_path=config.model_evaluator_path,
                          input_path=config.input_path)

    summary = get_summary()
    qcd_data = evaluator.get_qcd_test_data(summary=summary)
    qcd_loss = evaluator.get_error(qcd_data, summary=summary)
    
    signals_losses = []
    signals = get_signals()
    
    for name, path in signals.items():
        
        signal_data = evaluator.get_signal_test_data(name, path, summary)
        signal_loss = evaluator.get_error(signal_data, summary=summary)
        signals_losses.append(signal_loss)
    
    return qcd_loss, signals_losses


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