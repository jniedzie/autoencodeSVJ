import module.SummaryProcessor as summaryProcessor
from module.Evaluator import Evaluator
# from module.AutoEncoderEvaluator import AutoEncoderEvaluator
# from module.BdtEvaluator import BdtEvaluator
import importlib, argparse

# ------------------------------------------------------------------------------------------------
# This script will draw ROC curves for a specified model against all signals found in the
# "signals_base_path". If no version is specified (set to None), the latest training
# will be used.
# ------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("-c", "--config", dest="config_path", default=None, required=True, help="Path to the config file")
args = parser.parse_args()
config_path = args.config_path.strip(".py").replace("/", ".")
config = importlib.import_module(config_path)

# masses = [1500, 2000, 2500, 3000, 3500, 4000]
masses = [2000]
rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]

signals = {"{}, {}".format(mass, rinv): "{}{}GeV_{:1.2f}/base_3/*.h5".format(config.signals_base_path, mass, rinv)
           for mass in masses
           for rinv in rinvs}

if config.model_type == "AutoEncoder":

    evaluator = Evaluator(model_type=Evaluator.ModelTypes.AutoEncoder, input_path=config.input_path)
    evaluator.draw_roc_curves(summary_path=config.summary_path,
                              summary_version=config.best_model,
                              signals=signals,
                              xscale='linear')
    
elif config.model_type == "BDT":
    evaluator = Evaluator(model_type=Evaluator.ModelTypes.Bdt)
    evaluator.draw_roc_curves(summary_path=config.summary_path,
                              summary_version=config.best_model,
                              signals_base_path=config.signals_base_path,
                              xscale='linear')
else:
    print("Unknown model type: ", config.model_type)