import module.SummaryProcessor as summaryProcessor
from module.AutoEncoderEvaluator import AutoEncoderEvaluator
import importlib, argparse

# ------------------------------------------------------------------------------------------------
# This script will draw ROC curves for a specified model against all signals found in the
# "signals_base_path". If no version is specified (set to None), the latest training
# will be used.
# ------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("-c", "--config", dest="config_path", default=None, required=True, help="Path to the config file")
args = parser.parse_args()
config = importlib.import_module(args.config_path)


summary_base_name = "hlf_eflow{}_{}_".format(config.efp_base, config.target_dim)

input_summary_path = summaryProcessor.get_latest_summary_file_path(
    summaries_path=config.summary_path,
    file_name_base=summary_base_name,
    version=config.best_model)


# masses = [1500, 2000, 2500, 3000, 3500, 4000]
masses = [2000]
rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]

signals = {"{}, {}".format(mass, rinv) : "{}{}GeV_{:1.2f}/base_3/*.h5".format(config.signals_base_path, mass, rinv)
           for mass in masses
           for rinv in rinvs}

print("\n\nDraing ROC curves for summary: ", input_summary_path)

evaluator = AutoEncoderEvaluator(input_summary_path, signals=signals)
evaluator.roc(xscale='log', metrics=["mae"])