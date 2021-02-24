import module.SummaryProcessor as summaryProcessor
import importlib, argparse

# ------------------------------------------------------------------------------------------------
# This script will produce a CSV file with areas under ROC curves (AUCs) for each training
# summary file found in the "summary_path", testing on all signal samples found
# in "input_path" and store the output in "aucs_path", as defined in provided config file.
# ------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("-c", "--config", dest="config_path", default=None, required=True, help="Path to the config file")
args = parser.parse_args()
config = importlib.import_module(args.config_path)

summaryProcessor.save_all_missing_AUCs(
    summary_path=config.summary_path,
    signals_path=config.input_path,
    AUCs_path=config.AUCs_path
)
