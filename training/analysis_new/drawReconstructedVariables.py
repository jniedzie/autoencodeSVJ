import sys
sys.path.insert(1, "../")

from plottingHelpers import *

import module.SummaryProcessor as summaryProcessor
from module.Evaluator import Evaluator

import importlib, argparse
from pathlib import Path
from ROOT import TH1D, kGreen, kBlue, TCanvas, gApplication, gStyle, TLegend, kRed, gPad, kOrange

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


def get_plots_for_all_variables(data, color, suffix=""):
    plots = []
    
    for variable_name in data.keys():
        hist = get_histogram(data, variable_name, color, suffix)
        plots.append(hist)
    
    return plots


def get_qcd_and_signal_reco_plots(summary, evaluator):
    qcd_input_data = evaluator.get_qcd_data(summary=summary, test_data_only=False)
    qcd_reconstructed = evaluator.get_reconstruction(input_data=qcd_input_data, summary=summary)

    qcd_input_plots = get_plots_for_all_variables(qcd_input_data, color=config.qcd_input_color, suffix="QCD_input")
    qcd_reco_plots = get_plots_for_all_variables(qcd_reconstructed, color=config.qcd_reco_color, suffix="QCD_reco")

    scaler = qcd_input_data.scaler

    signals_input_plots = []
    signals_reco_plots = []
    
    for path in get_signal_paths(config):
        signal_input_data = evaluator.get_signal_data(name="", path=path, summary=summary, test_data_only=False)
        signal_reco_data = evaluator.get_reconstruction(input_data=signal_input_data, summary=summary, scaler=scaler)
        
        signal_input_plot = get_plots_for_all_variables(signal_input_data, color=config.signal_input_color, suffix="SVJ_input")
        signals_input_plots.append(signal_input_plot)

        signal_reco_plot = get_plots_for_all_variables(signal_reco_data, color=config.signal_reco_color, suffix="SVJ_reco")
        signals_reco_plots.append(signal_reco_plot)

    return qcd_input_plots, qcd_reco_plots, signals_input_plots, signals_reco_plots


def save_variable_distribution_plots(summary, evaluator):
    
    qcd_input_plots, qcd_reco_plots, signal_input_plots, signal_reco_plots = get_qcd_and_signal_reco_plots(summary, evaluator)
    
    # transpose:
    signal_input_plots = [list(x) for x in zip(*signal_input_plots)]
    signal_reco_plots = [list(x) for x in zip(*signal_reco_plots)]
    
    canvas = TCanvas("", "", 2000, 1000)
    canvas.Divide(5, 2)
    i_plot = 1
    
    legend = TLegend(0.5, 0.5, 0.9, 0.9)
    legend.AddEntry(qcd_input_plots[0], "QCD input", "l")
    legend.AddEntry(qcd_reco_plots[0], "QCD reco", "l")
    legend.AddEntry(signal_input_plots[0][0], "SVJ input", "l")
    legend.AddEntry(signal_reco_plots[0][0], "SVJ reco", "l")
    
    plots = zip(qcd_input_plots, qcd_reco_plots, signal_input_plots, signal_reco_plots)
    for qcd_input_hist, qcd_reco_hist, signal_input_hists, signal_reco_hists in plots:
        canvas.cd(i_plot)

        first_plot = True
        
        for hist in signal_input_hists:
            hist.DrawNormalized()
            first_plot = False
            
        for hist in signal_reco_hists:
            hist.DrawNormalized("same")
        
        qcd_input_hist.DrawNormalized("" if first_plot else "same")
        qcd_reco_hist.DrawNormalized("same")
        
        i_plot += 1
    
    legend.Draw()
    canvas.Update()
    filename = config.plots_path + summary.training_output_path.split("/")[-1]  + ".pdf"
    
    canvas.SaveAs(filename)
    saved_plots.append(filename)


def main():
    gStyle.SetOptStat(0)
    Path(config.plots_path).mkdir(parents=True, exist_ok=True)
    
    evaluator = Evaluator(**config.evaluation_general_settings, **config.evaluation_settings)
    summaries = summaryProcessor.get_summaries_from_path(config.summary_path)

    for _, summary in summaries.df.iterrows():
    
        filename = summary.training_output_path.split("/")[-1]
        if config.test_filename_pattern not in filename:
            continue
        
        save_variable_distribution_plots(summary, evaluator)

    print("\n\nthe following plots were created:\n")
    for path in saved_plots:
        print(path)

    if run_application:
        gApplication.Run()

if __name__ == "__main__":
    main()