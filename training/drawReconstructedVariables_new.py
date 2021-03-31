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
config_path = args.config_path.strip(".py").replace("/", ".")
config = importlib.import_module(config_path)

# filename_pattern = "efp_4_bottle_18_arch_42__42_loss"
filename_pattern = "hlf_efp_3_bottle_9_arch_42__42_loss_mean_absolute_error_batch_size_256_noChargedFraction_v"

save_plots = False

fraction_of_models_for_avg_chi2 = 0.8

qcd_input_color = kGreen
qcd_reco_color = kBlue
signal_input_color = kRed
signal_reco_color = kOrange

bins = {
    "Eta" : (-3.5, 3.5, 100),
    "Phi" : (-3.5, 3.5, 100),
    "Pt"  : (0, 2000, 100),
    "M"  : (0, 800, 100),
    "ChargedFraction"  : (0, 1, 100),
    "PTD"  : (0, 1, 100),
    "Axis2"  : (0, 0.2, 100),
    "eflow"  : (0, 1, 100),
}

# masses = [1500, 2000, 2500, 3000, 3500, 4000]
# masses = [2500, 3000, 3500, 4000]
masses = [3000]
# rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
rinvs = [0.45]


def get_binning_for_variable(variable_name):
    if "eflow" in variable_name:
        min = bins["eflow"][0]
        max = bins["eflow"][1]
        n_bins = bins["eflow"][2]
    else:
        min = bins[variable_name][0]
        max = bins[variable_name][1]
        n_bins = bins[variable_name][2]
    
    return n_bins, min, max


def initialize_histogram(variable_name, suffix=""):
    n_bins, min, max = get_binning_for_variable(variable_name)
    return TH1D(variable_name + suffix, variable_name + suffix, n_bins, min, max)


def initialize_input_and_reco_histograms(variable_name):
    n_bins, min, max = get_binning_for_variable(variable_name)
    hist_input = TH1D(variable_name, variable_name, n_bins, min, max)
    hist_reco = TH1D(variable_name + "reco", variable_name + "reco", n_bins, min, max)
    
    return hist_input, hist_reco


def fill_histogram(hist, data, variable_name):
    values = data[variable_name].tolist()
    for value in values:
        hist.Fill(value)


def get_histogram(data, variable_name, color=None):
    hist = initialize_histogram(variable_name)
    fill_histogram(hist, data, variable_name)
    if color is not None:
        hist.SetLineColor(color)
    return hist


def get_summary():
    filename = config.file_name + "_v" + str(config.best_model)
    summary_path = config.summary_path + filename + ".summary"
    print("Loading summary from path: ", summary_path)
    summary = summaryProcessor.get_summary_from_path(summary_path)
    for _, s in summary.df.iterrows():
        summary = s
    return summary


def get_hists_chi_2(hist_1, hist_2):
    return hist_1.Chi2Test(hist_2, "CHI2/NDF")


def get_signals():
    signals = {"{}, {}".format(mass, rinv): "{}{}GeV_{:1.2f}/base_{}/*.h5".format(config.signals_base_path, mass, rinv, config.efp_base)
               for mass in masses
               for rinv in rinvs}
    return signals


def initialize_and_fill_loss_histogram(data, name):
    hist = TH1D(name, name, 40, 0.0, 0.4)
    
    for value in data:
        hist.Fill(value)
    
    hist.SetLineColor(kRed if name != "qcd" else kBlue)
    
    return hist


def get_loss_histograms(background_data, signals_data):
    
    hist_background = initialize_and_fill_loss_histogram(background_data, "qcd")

    hists_signal = []
    for i, signal_data in enumerate(signals_data):
        hist_signal = initialize_and_fill_loss_histogram(signal_data, "signal"+str(i))
        hists_signal.append(hist_signal)
    
    return hist_background, hists_signal


def get_qcd_and_signal_losses(summary, evaluator):
    qcd_data = evaluator.get_qcd_test_data(summary=summary)
    qcd_loss = evaluator.get_error(qcd_data, summary=summary)

    scaler = qcd_data.scaler

    signals_losses = []
    signals = get_signals()

    for name, path in signals.items():
        signal_data = evaluator.get_signal_data(name, path, summary, test_data_only=True)
        signal_loss = evaluator.get_error(signal_data, summary=summary, scaler=scaler)
        signals_losses.append(signal_loss)

    return qcd_loss, signals_losses


def get_plots_for_all_variables(data, color):
    plots = []
    
    for variable_name in data.keys():
        hist = get_histogram(data=data, variable_name=variable_name, color=color)
        plots.append(hist)
    
    return plots


def get_qcd_chi2_per_variable(summary, evaluator):
    qcd_data = evaluator.get_qcd_test_data(summary=summary)
    qcd_reconstructed = evaluator.get_reconstruction(input_data=qcd_data, summary=summary)
    
    chi2_per_variable = {}
    
    for variable_name in qcd_data.keys():
        input_hist = get_histogram(qcd_data, variable_name)
        reco_hist = get_histogram(qcd_reconstructed, variable_name)
        chi2_per_variable[variable_name] = get_hists_chi_2(input_hist, reco_hist)
    
    return chi2_per_variable


def save_variable_distribution_plots(summary, evaluator):
    filename = summary.training_output_path.split("/")[-1]

    qcd_data = evaluator.get_qcd_test_data(summary=summary)
    qcd_reconstructed = evaluator.get_reconstruction(input_data=qcd_data, summary=summary)

    input_plots = get_plots_for_all_variables(qcd_data, color=qcd_input_color)
    reco_plots = get_plots_for_all_variables(qcd_reconstructed, color=qcd_reco_color)
    
    signals = get_signals()

    signals_plots = []
    for name, path in signals.items():
        signal_data = evaluator.get_signal_data(name, path, summary, test_data_only=False)
        signal_plot = get_plots_for_all_variables(signal_data, color=signal_input_color)
        signals_plots.append(signal_plot)

    signals_plots = [list(x) for x in zip(*signals_plots)]
    
    canvas = TCanvas("", "", 2000, 1000)
    canvas.Divide(5, 2)
    i_plot = 1
    
    legend = TLegend(0.5, 0.5, 0.9, 0.9)
    legend.AddEntry(input_plots[0], "Input", "l")
    legend.AddEntry(reco_plots[0], "Reco", "l")
    legend.AddEntry(signals_plots[0][0], "Signal", "l")
    
    for hist_input, hist_reco, hists_signal in zip(input_plots, reco_plots, signals_plots):
        canvas.cd(i_plot)
        
        first_plot = True
        
        for hist_signal in hists_signal:
            hist_signal.DrawNormalized()
            first_plot = False
        
        if first_plot:
            hist_input.DrawNormalized()
            first_plot = False
        else:
            hist_input.DrawNormalized("same")
        
        hist_reco.DrawNormalized("same")
        
        i_plot += 1
    
    legend.Draw()
    canvas.Update()
    canvas.SaveAs(config.plots_path + filename + ".pdf")


def get_reconstruction_chi2s(summaries, evaluator):
    chi2_values = {}
    
    for _, summary in summaries.df.iterrows():
        
        filename = summary.training_output_path.split("/")[-1]
        if filename_pattern not in filename:
            continue
        
        chi2_values[filename] = get_qcd_chi2_per_variable(summary, evaluator)
    
    return chi2_values


def get_ok_high_very_high_chi2_counts(chi2_values):
    
    ok_count = {}
    high_count = {}
    very_high_count = {}
    
    for filename, chi2s in chi2_values.items():
        
        chi2_sum = 0
        
        print("\n\nChi2's for file: ", filename)
        for name, chi2 in chi2s.items():
            chi2_sum += chi2
            
            if name not in very_high_count.keys():
                very_high_count[name] = 0
            if name not in high_count.keys():
                high_count[name] = 0
            if name not in ok_count.keys():
                ok_count[name] = 0
            
            print("chi2 for ", name, ":\t", chi2)
            if chi2 > 100:
                very_high_count[name] += 1
            elif chi2 > 10:
                high_count[name] += 1
            else:
                ok_count[name] += 1
    
    return ok_count, high_count, very_high_count


def get_n_fully_and_partially_successful(chi2_values):
    fully_successful_count = 0
    partially_successful_count = 0
    
    for filename, chi2s in chi2_values.items():
        
        is_fully_successful = True
        is_partially_successful = True
        
        for name, chi2 in chi2s.items():
            if chi2 > 100:
                is_fully_successful = False
                is_partially_successful = False
            elif chi2 > 10:
                is_fully_successful = False
        
        if is_fully_successful:
            fully_successful_count += 1
        if is_partially_successful:
            partially_successful_count += 1
            
    return fully_successful_count, partially_successful_count


def get_total_chi2(chi2_values):
    total_chi2 = {}
    
    for filename, chi2s in chi2_values.items():
        chi2_sum = 0
        
        for name, chi2 in chi2s.items():
            chi2_sum += chi2
        
        total_chi2[filename] = chi2_sum
    
    return total_chi2


def save_loss_plots(summaries, evaluator):
    canvas = TCanvas("", "", 2880, 1800)
    canvas.Divide(5, 2)
    i_plot = 1
    
    for _, summary in summaries.df.iterrows():
        
        filename = summary.training_output_path.split("/")[-1]
        if filename_pattern not in filename:
            continue

        qcd_loss, signals_losses = get_qcd_and_signal_losses(summary, evaluator)
        hist_background, hists_signal = get_loss_histograms(qcd_loss, signals_losses)
        
        canvas.cd(i_plot)
        
        legend = TLegend(0.5, 0.5, 0.9, 0.9)
        legend.AddEntry(hist_background, "Background", "l")
        legend.AddEntry(hists_signal[0], "Signal", "l")
        
        version = summaryProcessor.get_version(summary.summary_path)
        
        hist_background.SetTitle("v" + str(version))
        hist_background.GetXaxis().SetTitle("loss")
        
        hist_background.DrawNormalized()
        
        for hist_signal in hists_signal:
            hist_signal.DrawNormalized("same")
        
        i_plot += 1
        
        legend.Draw()
    
    canvas.Update()
    canvas.SaveAs(config.plots_path + "losses_" + filename_pattern + ".pdf")
    
    for i in range(i_plot - 1):
        canvas.cd(i + 1)
        gPad.SetLogy()
    
    canvas.Update()
    canvas.SaveAs(config.plots_path + "losses_" + filename_pattern + "_log.pdf")
    
    
def print_summary(ok_count, high_count, very_high_count,
                  fully_successful_count, partially_successful_count,
                  total_chi2):
    print("\n\n Summary: \n")
    print("Variable\tok\thigh\tvery high")
    
    for name in ok_count.keys():
        print(name, "\t", ok_count[name], "\t", high_count[name], "\t", very_high_count[name])
    
    print("\n\nFully successful: ", fully_successful_count)
    print("Partially successful: ", partially_successful_count)
    print("Out of: ", len(total_chi2))
    
    print("\n\nTotal chi2's per file:")
    
    average_chi2 = 0

    total_chi2_sorted = {k: v for k, v in sorted(total_chi2.items(), key=lambda item: item[1])}
    n_models_to_check = len(total_chi2) * fraction_of_models_for_avg_chi2
    
    i_model = 0
    for filename, chi2 in total_chi2_sorted.items():
        print(filename, "\t:", chi2)
        
        if i_model < n_models_to_check:
            print("included")
            average_chi2 += chi2
            
        i_model += 1
    
    print("Average total chi2: ", average_chi2 / len(total_chi2))


def main():
    gStyle.SetOptStat(0)
    Path(config.plots_path).mkdir(parents=True, exist_ok=True)
    
    evaluator = Evaluator(model_evaluator_path=config.model_evaluator_path,
                          input_path=config.input_path)
    
    summaries = summaryProcessor.get_summaries_from_path(config.summary_path)
   
    if save_plots:
        save_loss_plots(summaries, evaluator)
        
        for _, summary in summaries.df.iterrows():
        
            filename = summary.training_output_path.split("/")[-1]
            if filename_pattern not in filename:
                continue
        
            save_variable_distribution_plots(summary, evaluator)

    chi2_values = get_reconstruction_chi2s(summaries, evaluator)

    ok_count, high_count, very_high_count = get_ok_high_very_high_chi2_counts(chi2_values)
    fully_successful_count, partially_successful_count = get_n_fully_and_partially_successful(chi2_values)
    total_chi2 = get_total_chi2(chi2_values)

    print_summary(ok_count, high_count, very_high_count, fully_successful_count, partially_successful_count, total_chi2)
    
    # gApplication.Run()

if __name__ == "__main__":
    main()