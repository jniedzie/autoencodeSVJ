from module.AutoEncoderTrainer import AutoEncoderTrainer
import importlib, argparse

# ------------------------------------------------------------------------------------------------
# This script will run N auto-encoder trainings with hyper-parameters, input data specified,
# output/summary paths and scaling as defined in the config file provided as an argument.
# ------------------------------------------------------------------------------------------------


parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("-c", "--config", dest="config_path", default=None, required=True, help="Path to the config file")
args = parser.parse_args()
config = importlib.import_module(args.config_path)

for i in range(config.n_models):
        
    trainer = AutoEncoderTrainer(qcd_path=config.qcd_path,
                                 bottleneck_size=config.target_dim,
                                 training_params=config.training_params,
                                 norm_type=config.norm_type,
                                 norm_args=config.norm_args
                                 )
    
    trainer.run_training(training_output_path=config.results_path,
                         summaries_path=config.summary_path,
                         verbose=True
                         )
    
    trainer.save_last_training_summary(path=config.summary_path)
        
    print('model {} finished'.format(i))