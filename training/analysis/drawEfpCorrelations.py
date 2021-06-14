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


def get_plots_for_all_efp_combinations(data, suffix=""):
    plots = [[]]

    efp_variables = [int(x.replace("efp ", "")) for x in data.keys() if "efp" in x]
    efp_variables.sort()

    for efp_x in efp_variables:
        print("Preparing hists for efp: ", efp_x)
        plots_row = []

        for efp_y in efp_variables:
            hist = get_histogram_2d(data, "efp "+str(efp_x), "efp "+str(efp_y), suffix)
            plots_row.append(hist)
        plots.append(plots_row)
    
    return plots


def drop_efp_variables(variables, min):
    new_variables = []
    for var in variables:
        if "efp" in var:
            efp_number = int(var.replace("efp ", ""))
            if efp_number < min:
                new_variables.append(var)
        else:
            new_variables.append(var)
    return new_variables


def get_plots_for_all_variable_combinations(data, suffix=""):
    plots = [[]]

    variables_to_plot = drop_efp_variables(data.keys(), min=2)

    for var_x in variables_to_plot:
        print("Preparing hists for: ", var_x)
        plots_row = []

        for var_y in variables_to_plot:
            hist = get_histogram_2d(data, var_x, var_y, suffix)
            plots_row.append(hist)
        plots.append(plots_row)

    return plots


def get_qcd_efp_plots(summary, evaluator):
    qcd_input_data = evaluator.get_qcd_data(summary=summary, test_data_only=False)

    # path = get_signal_paths(config)[0]
    # qcd_input_data = evaluator.get_signal_data(name="", path=path, summary=summary, test_data_only=False)

    # qcd_input_plots = get_plots_for_all_efp_combinations(qcd_input_data, suffix="QCD_input")
    qcd_input_plots = get_plots_for_all_variable_combinations(qcd_input_data, suffix="QCD_input")

    return qcd_input_plots


def save_efp_correlation_plots(summary, evaluator):
    
    qcd_plots = get_qcd_efp_plots(summary, evaluator)

    n_rows = len(qcd_plots)

    canvas = TCanvas("", "", 2000, 1000)
    canvas.Divide(n_rows-1, n_rows-1)
    i_plot = 1

    for plots_row in qcd_plots:
        for plot in plots_row:
            canvas.cd(i_plot)
            plot.Draw("colz")
            i_plot += 1

    canvas.Update()
    filename = config.plots_path + summary.training_output_path.split("/")[-1]  + "_efp_corr.pdf"
    
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
        
        save_efp_correlation_plots(summary, evaluator)
        break

    print("\n\nthe following plots were created:\n")
    for path in saved_plots:
        print(path)

    if run_application:
        gApplication.Run()

if __name__ == "__main__":
    main()