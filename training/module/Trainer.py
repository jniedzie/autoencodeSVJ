from module.DataProcessor import DataProcessor
from module.DataLoader import DataLoader
import module.SummaryProcessor as summaryProcessor
import module.utils as utils

import numpy as np


class Trainer:

    def __init__(self, model_trainer_path, validation_data_fraction, test_data_fraction,
                 include_hlf, include_efp, hlf_to_drop, **kwargs):
    
        self.model_class = utils.import_class(model_trainer_path)
        
        self.seed = np.random.randint(0, 99999999)
        self.validation_data_fraction = validation_data_fraction
        self.test_data_fraction = test_data_fraction
        self.include_hlf = include_hlf
        self.include_efp = include_efp
        self.hlf_to_drop = hlf_to_drop

        data_processor = DataProcessor(validation_fraction=validation_data_fraction,
                                       test_fraction=test_data_fraction,
                                       seed=self.seed)

        data_loader = DataLoader()
        data_loader.set_params(include_hlf=include_hlf, include_eflow=include_efp, hlf_to_drop=hlf_to_drop)

        utils.set_random_seed(self.seed)
        self.model_trainer = self.model_class(data_processor=data_processor,
                                              data_loader=data_loader,
                                              **kwargs)
        
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
            'include_hlf': self.include_hlf,
            'include_efp': self.include_efp,
            'hlf_to_drop': tuple(self.hlf_to_drop),
        }
        
        return summary_dict
