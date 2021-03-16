from module.DataProcessor import DataProcessor
import module.SummaryProcessor as summaryProcessor
import module.utils as utils

from enum import Enum
import numpy as np
import sys
import importlib


class Trainer:
    class ModelTypes(Enum):
        AutoEncoder = 0
        Bdt = 1

    ModelTypeNames = {
        ModelTypes.AutoEncoder: "AutoEncoder",
        ModelTypes.Bdt: "BDT",
    }

    def __init__(self, model_trainer_path, validation_data_fraction, test_data_fraction , **kwargs):
    
        self.model_class = utils.import_class(model_trainer_path)
        
        self.seed = np.random.randint(0, 99999999)
        self.validation_data_fraction = validation_data_fraction
        self.test_data_fraction = test_data_fraction

        data_processor = DataProcessor(validation_fraction=validation_data_fraction,
                                       test_fraction=test_data_fraction,
                                       seed=self.seed)

        utils.set_random_seed(self.seed)
        self.model_trainer = self.model_class(data_processor=data_processor, **kwargs)
        
    def train(self, summaries_path):
        
        self.model_trainer.train()

        summary_dict = self.model_trainer.get_summary()
        summary_dict = {**summary_dict, **self.__get_summary()}
        summaryProcessor.dump_summary_json(summary_dict, output_path=summaries_path)
        
    def __get_summary(self):
        
        summary_dict = {
            "model_type": str(self.model_class),
            "seed": self.seed,
            'val_split': self.validation_data_fraction,
            'test_split': self.test_data_fraction,
        }
        
        return summary_dict
