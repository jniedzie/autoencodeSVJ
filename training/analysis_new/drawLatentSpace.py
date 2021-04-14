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
        signal_input_data = evaluator.get_signal_data(name="", path=path, summary=summary, test_data_only=False)
        signal_reco_data = evaluator.get_latent_space_values(input_data=signal_input_data, summary=summary, scaler=scaler)

        signal_latent_plot = get_plots_for_all_variables(signal_reco_data, color=config.signal_reco_color, suffix="SVJ")
        signals_latent_plots.append(signal_latent_plot)

    return qcd_latent_plots, signals_latent_plots, qcd_latent_correlation_plots


def save_1d_latent_plots(summary, evaluator):
    
    qcd_latent_plots, signal_latent_plots, qcd_latent_correlation_plots = get_qcd_and_signal_reco_plots(summary, evaluator)
    
    # transpose:
    signal_latent_plots = [list(x) for x in zip(*signal_latent_plots)]
    
    canvas = TCanvas("", "", 2000, 1000)
    canvas.Divide(5, 2)
    i_plot = 1
    
    legend = TLegend(0.5, 0.5, 0.9, 0.9)
    legend.AddEntry(qcd_latent_plots[0], "QCD latent", "l")
    legend.AddEntry(signal_latent_plots[0][0], "SVJ latent", "l")
    
    plots = zip(qcd_latent_plots, signal_latent_plots)

    total_chi2 = 0

    for qcd_latent_plot, signal_latent_plots in plots:
        canvas.cd(i_plot)

        chi2 = get_hists_chi_2(signal_latent_plots[0], qcd_latent_plot)
        print("chi2 ", i_plot - 1, " :", chi2)
        total_chi2 += chi2

        first_plot = True
        
        for hist in signal_latent_plots:
            hist.DrawNormalized()
            first_plot = False

        qcd_latent_plot.DrawNormalized("" if first_plot else "same")



        i_plot += 1

    print("total chi2: ", total_chi2)

    legend.Draw()
    canvas.Update()
    filename = config.plots_path + summary.training_output_path.split("/")[-1]  + ".pdf"
    
    canvas.SaveAs(filename)
    saved_plots.append(filename)


    canvas_correlation = TCanvas("corr", "corr", 2000, 1000)
    n_rows = len(qcd_latent_correlation_plots)
    canvas_correlation.Divide(n_rows-1, n_rows-1)

    i_plot = 1

    avg_correlation = 0
    n_avg_correlation = 0

    for x, plots_row in enumerate(qcd_latent_correlation_plots):

        for y, plot in enumerate(plots_row):
            canvas_correlation.cd(i_plot)
            plot.Draw("colz")

            correlation = plot.GetCorrelationFactor()

            print(correlation, end="\t")

            if x < y:
                avg_correlation += correlation
                n_avg_correlation += 1

            i_plot += 1

        print()


    avg_correlation = avg_correlation/n_avg_correlation

    print("Avg correlation: ", avg_correlation)

    canvas_correlation.Update()
    filename = config.plots_path + summary.training_output_path.split("/")[-1] + "_latent_corr.pdf"

    canvas_correlation.SaveAs(filename)
    saved_plots.append(filename)



def main():
    gStyle.SetOptStat(0)
    Path(config.plots_path).mkdir(parents=True, exist_ok=True)
    
    evaluator = Evaluator(**config.evaluation_general_settings, **config.evaluation_settings)
    summaries = summaryProcessor.get_summaries_from_path(config.summary_path)

    print("Looking for file matching: ", config.test_filename_pattern)

    for _, summary in summaries.df.iterrows():
    
        filename = summary.training_output_path.split("/")[-1]

        print("File name:", filename)

        if config.test_filename_pattern not in filename:
            continue
        
        save_1d_latent_plots(summary, evaluator)
        break

    print("\n\nthe following plots were created:\n")
    for path in saved_plots:
        print(path)

    if run_application:
        gApplication.Run()

if __name__ == "__main__":
    main()