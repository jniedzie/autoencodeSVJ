from module.Evaluator import Evaluator
import importlib, argparse

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


if config.model_type == "AutoEncoder":
    evaluator = Evaluator(model_evaluator_path=config.model_evaluator_path,
                          input_path=config.input_path
                          )
    evaluator.save_aucs(summary_path=config.summary_path,
                        AUCs_path = config.AUCs_path
                        )
    
elif config.model_type == "BDT":
    evaluator = Evaluator(model_evaluator_path=config.model_evaluator_path)

    evaluator.save_aucs(summary_path=config.summary_path,
                        AUCs_path=config.AUCs_path,
                        signals_base_path=config.signals_base_path
                        )
else:
    print("Unknown model type: ", config.model_type)