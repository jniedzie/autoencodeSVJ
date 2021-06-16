import importlib
import argparse

from module.Evaluator import Evaluator

# ------------------------------------------------------------------------------------------------
# This script will produce a CSV file with areas under ROC curves (AUCs) for each training
# summary file found in the "summary_path", testing on all signal samples found
# in "input_path" and store the output in "aucs_path", as defined in provided config file.
# ------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("-c", "--config", dest="config_path", default=None, required=True, help="Path to the config file")
args = parser.parse_args()
config_path = args.config_path.strip(".py").replace("/", ".")
config = importlib.import_module(config_path)


evaluator = Evaluator(**config.evaluation_general_settings, **config.evaluation_settings)
evaluator.save_aucs(test_filename_pattern=config.test_filename_pattern, use_qcd_weights_for_signal=True)
# evaluator.save_aucs(test_filename_pattern="v", use_qcd_weights_for_signal=True)
