from module.DataProcessor import DataProcessor
from module.DataLoader import DataLoader
import module.SummaryProcessor as summaryProcessor
import module.utils as utils

import numpy as np
import datetime
from pathlib import Path
import os, pickle


class Trainer:

    def __init__(self,
                 # general settings of the training
                 model_trainer_path,
                 validation_data_fraction,
                 test_data_fraction,
                 include_hlf,
                 include_efp,
                 hlf_to_drop,
                 # arguments that will be passed to the specialized trainer class
                 **training_settings):
    
        self.model_class = utils.import_class(model_trainer_path)
        
        self.seed = np.random.randint(0, 99999999)
        self.validation_data_fraction = validation_data_fraction
        self.test_data_fraction = test_data_fraction
        self.include_hlf = include_hlf
        self.include_efp = include_efp
        self.hlf_to_drop = hlf_to_drop

        self.training_output_path = training_settings["training_output_path"]

        data_processor = DataProcessor(validation_fraction=validation_data_fraction,
                                       test_fraction=test_data_fraction,
                                       seed=self.seed)

        data_loader = DataLoader()
        data_loader.set_params(include_hlf=include_hlf, include_eflow=include_efp, hlf_to_drop=hlf_to_drop)

        utils.set_random_seed(self.seed)
        self.model_trainer = self.model_class(data_processor=data_processor,
                                              data_loader=data_loader,
                                              **training_settings)
        
    def train(self, summaries_path):
        
        print("\n\nTraining the model")
        self.start_timestamp = datetime.datetime.now()
        self.model_trainer.train()
        self.end_timestamp = datetime.datetime.now()
        print("Training executed in: ", (self.end_timestamp - self.start_timestamp), " s")

        self.__save_model()

        summary_dict = self.model_trainer.get_summary()
        summary_dict = {**summary_dict, **self.__get_summary()}
        summaryProcessor.dump_summary_json(summary_dict, output_path=summaries_path)
      
    def __save_model(self):
        model = self.model_trainer.model

        pickle_output_path = self.training_output_path + ".pkl"
        Path(os.path.dirname(pickle_output_path)).mkdir(parents=True, exist_ok=True)
        
        try:
            print("Trying to save model using to_json")
            pickle.dump(model.to_json(), open(pickle_output_path, 'wb'))
        except AttributeError:
            print("Failed saving model using to_json")
            try:
                print("Trying to save model directly")
                pickle.dump(model, open(pickle_output_path, 'wb'))
            except TypeError:
                print("Failed to save model")

        print("Successfully save the model")
        
    def __get_summary(self):
        
        summary_dict = {
            "model_type": str(self.model_class),
            "seed": self.seed,
            'val_split': self.validation_data_fraction,
            'test_split': self.test_data_fraction,
            'include_hlf': self.include_hlf,
            'include_efp': self.include_efp,
            'hlf_to_drop': tuple(self.hlf_to_drop),
            'start_time': str(self.start_timestamp),
            'end_time': str(self.end_timestamp),
        }
        
        return summary_dict
