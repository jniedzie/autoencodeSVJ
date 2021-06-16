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

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("-c", "--config", dest="config_path", default=None, required=True, help="Path to the config file")
args = parser.parse_args()
config_path = args.config_path.replace(".py", "").replace("../", "").replace("/", ".")
config = importlib.import_module(config_path)


def get_plots_for_all_variables(data, color, suffix=""):
    plots = []
    
    for variable_name in data.keys():
        hist = get_histogram(data, "latent_"+str(variable_name), color, suffix)
        plots.append(hist)
    
    return plots


def get_plots_for_all_variable_combinations(data, suffix=""):
    plots = [[]]

    for var_x in data.keys():
        print("Preparing hists for: ", var_x)
        plots_row = []
        var_x_name = "latent_"+str(var_x)

        for var_y in data.keys():
            var_y_name = "latent_" + str(var_y)

            hist = get_histogram_2d(data, var_x_name, var_y_name, suffix)
            plots_row.append(hist)
        plots.append(plots_row)

    return plots


def get_qcd_and_signal_reco_plots(summary, evaluator):
    qcd_input_data = evaluator.get_qcd_data(summary=summary, test_data_only=False)
    qcd_latent_values = evaluator.get_latent_space_values(input_data=qcd_input_data, summary=summary)

    qcd_latent_plots = get_plots_for_all_variables(qcd_latent_values, color=config.qcd_reco_color, suffix="QCD")
    qcd_latent_correlation_plots = get_plots_for_all_variable_combinations(qcd_latent_values, suffix="QCD")

    scaler = qcd_input_data.scaler

    signals_latent_plots = []

    for path in get_signal_paths(config):
        signal_input_data = evaluator.get_signal_data(path=path, summary=summary, test_data_only=False)
        signal_reco_data = evaluator.get_latent_space_values(input_data=signal_input_data, summary=summary, scaler=scaler)

        signal_latent_plot = get_plots_for_all_variables(signal_reco_data, color=config.signal_reco_color, suffix="SVJ")
        signals_latent_plots.append(signal_latent_plot)

    return qcd_latent_plots, signals_latent_plots, qcd_latent_correlation_plots


def get_chi2_and_correlation(summary, evaluator):
    
    qcd_latent_plots, signal_latent_plots, qcd_latent_correlation_plots = get_qcd_and_signal_reco_plots(summary, evaluator)
    
    # transpose:
    signal_latent_plots = [list(x) for x in zip(*signal_latent_plots)]

    total_chi2 = 0
    for qcd_latent_plot, signal_latent_plots in zip(qcd_latent_plots, signal_latent_plots):
        chi2 = get_hists_chi_2(signal_latent_plots[0], qcd_latent_plot)
        total_chi2 += chi2

    avg_correlation = 0
    n_avg_correlation = 0

    for x, plots_row in enumerate(qcd_latent_correlation_plots):
        for y, plot in enumerate(plots_row):
            correlation = plot.GetCorrelationFactor()
            if x < y:
                avg_correlation += correlation
                n_avg_correlation += 1

    avg_correlation = avg_correlation/n_avg_correlation

    return total_chi2, avg_correlation


def main():
    gStyle.SetOptStat(0)
    Path(config.plots_path).mkdir(parents=True, exist_ok=True)
    
    evaluator = Evaluator(**config.evaluation_general_settings, **config.evaluation_settings)
    summaries = summaryProcessor.get_summaries_from_path(config.summary_path)

    print("Looking for file matching: ", config.test_filename_pattern)

    total_chi2s_and_avg_correlations = {}

    total_chi2s = []
    avg_correletaions = []

    for _, summary in summaries.df.iterrows():
    
        filename = summary.training_output_path.split("/")[-1]

        print("File name:", filename)

        if config.test_filename_pattern not in filename:
            continue
        
        total_chi2, avg_correlation = get_chi2_and_correlation(summary, evaluator)
        total_chi2s_and_avg_correlations[filename] = (total_chi2, avg_correlation)

        total_chi2s.append(total_chi2)
        avg_correletaions.append(avg_correlation)

    average_chi2 = 0
    average_correlation = 0

    total_chi2s = sorted(total_chi2s, reverse=True)
    avg_correletaions = [abs(i) for i in avg_correletaions]
    avg_correletaions = sorted(avg_correletaions)

    n_models_to_check = len(total_chi2s) * config.fraction_of_models_for_avg_chi2

    i_model = 0
    for chi2, corr in zip(total_chi2s, avg_correletaions):
        print("chi2: ", chi2)


        if i_model < n_models_to_check:
            print("included")
            average_chi2 += chi2
            average_correlation += corr

        else:
            print("excluded")

        i_model += 1


    average_chi2 /= n_models_to_check
    average_correlation /= n_models_to_check

    print("Average chi2 (x% best): ", average_chi2)
    print("Average correlation (x% best): ", average_correlation)

    for filename, params in total_chi2s_and_avg_correlations.items():
        print("File: ", filename)
        print("Total chi2 and average correlation: ", params)



if __name__ == "__main__":
    main()