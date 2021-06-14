import sys
sys.path.insert(1, "../")

from plottingHelpers import *

import module.SummaryProcessor as summaryProcessor
from module.Evaluator import Evaluator

import importlib, argparse
from pathlib import Path
from ROOT import TCanvas, gApplication, gStyle, TLegend, gPad

# ------------------------------------------------------------------------------------------------
# This script will draw input and reconstructed variables for signal found in the
# "signals_base_path" and background as specified in the training summary.
# ------------------------------------------------------------------------------------------------

run_application = False

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("-c", "--config", dest="config_path", default=None, required=True, help="Path to the config file")
args = parser.parse_args()
config_path = args.config_path.replace(".py", "").replace("../", "").replace("/", ".")
config = importlib.import_module(config_path)

saved_plots = []


def get_loss_histograms(background_data, signals_data, suffix="", qcd_weights=None):
    hist_background = get_histogram(background_data, "loss", config.qcd_input_color, suffix="QCD_"+suffix, weights=qcd_weights)
    
    hists_signal = []
    for i, signal_data in enumerate(signals_data):
        hist_signal = get_histogram(signal_data, "loss", config.signal_input_color, suffix="SVJ_"+str(i)+"_"+suffix)
        hists_signal.append(hist_signal)
    
    return hist_background, hists_signal


def get_qcd_and_signal_losses(summary, evaluator):
    qcd_data = evaluator.get_qcd_data(summary=summary, test_data_only=True)
    qcd_loss = evaluator.get_error(qcd_data, summary=summary)
    qcd_weights = qcd_data.weights
    
    scaler = qcd_data.scaler

    signals_losses = []
    
    for path in get_signal_paths(config):
        signal_data = evaluator.get_signal_data(name="", path=path, summary=summary, test_data_only=False)
        signal_loss = evaluator.get_error(signal_data, summary=summary, scaler=scaler)
        signals_losses.append(signal_loss)
    
    return qcd_loss, signals_losses, qcd_weights


def draw_plots(summary, evaluator):
    version = summaryProcessor.get_version(summary.summary_path)
    
    qcd_loss, signals_losses, qcd_weights = get_qcd_and_signal_losses(summary, evaluator)
    
    hist_background, hists_signal = get_loss_histograms(qcd_loss, signals_losses, suffix="v"+str(version), qcd_weights=qcd_weights)
    
    hist_background.SetTitle("v" + str(version))
    hist_background.GetXaxis().SetTitle("loss")
    
    hist_background.DrawNormalized()
    
    for hist_signal in hists_signal:
        hist_signal.DrawNormalized("same")
    
    return hist_background, hists_signal[0]


def save_canvas(canvas, legend, n_plots):
    legend.Draw()
    
    canvas.Update()
    filename = config.plots_path + "losses_" + config.test_filename_pattern + ".pdf"
    canvas.SaveAs(filename)
    saved_plots.append(filename)
    
    for i in range(n_plots - 1):
        canvas.cd(i + 1)
        gPad.SetLogy()
    
    canvas.Update()
    filename = config.plots_path + "losses_" + config.test_filename_pattern + "_log.pdf"
    canvas.SaveAs(filename)
    saved_plots.append(filename)


def main():
    gStyle.SetOptStat(0)
    Path(config.plots_path).mkdir(parents=True, exist_ok=True)
    
    evaluator = Evaluator(**config.evaluation_general_settings, **config.evaluation_settings)
    summaries = summaryProcessor.get_summaries_from_path(config.evaluation_general_settings["summary_path"])
    
    i_plot = 1
    canvas = TCanvas("", "", 2880, 1800)
    canvas.Divide(5, 2)
    
    background, signal = None, None
    
    for _, summary in summaries.df.iterrows():
        
        filename = summary.training_output_path.split("/")[-1]
        if config.test_filename_pattern not in filename:
            continue
            
        canvas.cd(i_plot)
        background, signal = draw_plots(summary, evaluator)
        
        i_plot += 1

    legend = TLegend(0.5, 0.5, 0.9, 0.9)
    legend.AddEntry(background, "Background", "l")
    legend.AddEntry(signal, "Signal", "l")
    
    save_canvas(canvas, legend, i_plot)
    
    print("\n\nthe following plots were created:\n")
    for path in saved_plots:
        print(path)
    
    if run_application:
        gApplication.Run()


if __name__ == "__main__":
    main()