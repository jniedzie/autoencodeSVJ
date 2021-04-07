import sys
sys.path.insert(1, "../")

from plottingHelpers import *

import module.SummaryProcessor as summaryProcessor
from module.Evaluator import Evaluator

import importlib, argparse

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("-c", "--config", dest="config_path", default=None, required=True, help="Path to the config file")
args = parser.parse_args()
config_path = args.config_path.replace(".py", "").replace("../", "").replace("/", ".")
config = importlib.import_module(config_path)


def get_qcd_chi2_per_variable(summary, evaluator):
    qcd_input_data = evaluator.get_qcd_data(summary=summary, test_data_only=True)
    qcd_reconstructed = evaluator.get_reconstruction(input_data=qcd_input_data, summary=summary)
    
    chi2_per_variable = {}
    
    for variable_name in qcd_input_data.keys():
        input_hist = get_histogram(qcd_input_data, variable_name)
        reco_hist = get_histogram(qcd_reconstructed, variable_name)
        chi2_per_variable[variable_name] = get_hists_chi_2(input_hist, reco_hist)
    
    return chi2_per_variable


def get_reconstruction_chi2s(summaries, evaluator):
    chi2_values = {}
    
    for _, summary in summaries.df.iterrows():
        
        filename = summary.training_output_path.split("/")[-1]
        if config.test_filename_pattern not in filename:
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
    n_models_to_check = len(total_chi2) * config.fraction_of_models_for_avg_chi2
    
    i_model = 0
    for filename, chi2 in total_chi2_sorted.items():
        print(filename, "\t:", chi2)
        
        if i_model < n_models_to_check:
            print("included")
            average_chi2 += chi2
        
        i_model += 1
    
    print("Average total chi2: ", average_chi2 / len(total_chi2))


def main():
    evaluator = Evaluator(**config.evaluation_general_settings, **config.evaluation_settings)
    summaries = summaryProcessor.get_summaries_from_path(config.summary_path)

    chi2_values = get_reconstruction_chi2s(summaries, evaluator)

    ok_count, high_count, very_high_count = get_ok_high_very_high_chi2_counts(chi2_values)
    fully_successful_count, partially_successful_count = get_n_fully_and_partially_successful(chi2_values)
    total_chi2 = get_total_chi2(chi2_values)

    print_summary(ok_count, high_count, very_high_count, fully_successful_count, partially_successful_count, total_chi2)


if __name__ == "__main__":
    main()
