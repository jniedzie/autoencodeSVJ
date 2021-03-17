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

for i in range(config.n_models):

    trainer = None

    if config.model_type == "AutoEncoder":
    
        file_name = config.file_name
        last_version = summaryProcessor.get_last_summary_file_version(config.summary_path, file_name)
        file_name += "_v{}".format(last_version + 1)
        
        trainer = Trainer(
            # general settings of the training
            model_trainer_path=config.model_trainer_path,
            validation_data_fraction=config.validation_data_fraction,
            test_data_fraction=config.test_data_fraction,
            include_hlf=config.include_hlf,
            include_efp=config.include_efp,
            hlf_to_drop=config.hlf_to_drop,
    
            # parameters that will be passed to your TrainerXYZ
            qcd_path=config.qcd_path,
            training_output_path=config.results_path+file_name,
            training_params=config.training_params,
            EFP_base=config.efp_base,
            norm_type=config.norm_type,
            norm_args=config.norm_args,
        )
        
    elif config.model_type == "BDT":
    
        i_sample = args.i_sample
    
        masses = [1500, 2000, 2500, 3000, 3500, 4000]
        rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
    
        i_mass = i_sample % len(masses)
        i_rinv = int(i_sample / len(masses))
    
        signal_name = "{}GeV_{:3.2f}".format(masses[i_mass], rinvs[i_rinv])
        signal_path = config.signals_base_path + "/" + signal_name + "/base_3/*.h5"

        file_name = config.file_name + "_" + signal_name
        last_version = summaryProcessor.get_last_summary_file_version(config.summary_path, file_name)

        file_name += "_v{}".format(last_version + 1)
        model_output_path = config.results_path + "/" + file_name + ".weigths"
        
        trainer = Trainer(
            # general settings of the training
            model_trainer_path=config.model_trainer_path,
            validation_data_fraction=config.validation_data_fraction,
            test_data_fraction=config.test_data_fraction,
            include_hlf=config.include_hlf,
            include_efp=config.include_efp,
            hlf_to_drop=config.hlf_to_drop,
            
            # parameters that will be passed to your TrainerXYZ
            qcd_path=config.qcd_path,
            signal_path=signal_path,
            training_output_path=config.results_path+file_name,
            training_params=config.training_params,
            EFP_base=config.efp_base,
            norm_type=config.norm_type,
            norm_args=config.norm_args,
        )
    else:
        print("Unrecognized model: ", config.model_type)
        exit(0)
    
    
    trainer.train(summaries_path=config.summary_path)
    
    
        
    print('model {} finished'.format(i))