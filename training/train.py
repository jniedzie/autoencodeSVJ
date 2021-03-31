import module.SummaryProcessor as summaryProcessor
from module.Trainer import Trainer

import importlib, argparse

# ------------------------------------------------------------------------------------------------
# This script will run N auto-encoder trainings with hyper-parameters, input data specified,
# output/summary paths and scaling as defined in the config file provided as an argument.
# ------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("-c", "--config", dest="config_path", default=None, required=True, help="Path to the config file")
parser.add_argument("-i", "--i_sample", dest="i_sample", default=0, type=int)
args = parser.parse_args()
config_path = args.config_path.strip(".py").replace("/", ".")
config = importlib.import_module(config_path)

from pathlib import Path
Path(config.output_path).mkdir(parents=True, exist_ok=True)
Path(config.summary_path).mkdir(parents=True, exist_ok=True)
Path(config.results_path).mkdir(parents=True, exist_ok=True)
Path(config.AUCs_path).mkdir(parents=True, exist_ok=True)

training_setting = config.training_settings


def get_sigal_name_for_index(i_sample):
    masses = [1500, 2000, 2500, 3000, 3500, 4000]
    rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
    
    i_mass = i_sample % len(masses)
    i_rinv = int(i_sample / len(masses))
    
    signal_name = "{}GeV_{:3.2f}".format(masses[i_mass], rinvs[i_rinv])
    
    return signal_name


for i in range(config.n_models):

    file_name = config.file_name
    
    if config.train_on_signal:
    
        signal_name = get_sigal_name_for_index(args.i_sample)
        signal_path = config.signals_base_path + "/" + signal_name + "/base_{}/*.h5".format(config.efp_base)
        training_setting["signal_path"] = signal_path
        
        file_name += "_" + signal_name

    last_version = summaryProcessor.get_last_summary_file_version(config.summary_path, file_name)
    file_name += "_v{}".format(last_version + 1)
    training_setting["training_output_path"] = config.results_path + file_name

    trainer = Trainer(
        **config.training_general_settings,
        **training_setting
    )

    trainer.train(summaries_path=config.summary_path)

    print('model {} finished'.format(i))