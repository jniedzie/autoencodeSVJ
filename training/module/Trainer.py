from enum import Enum
import numpy as np

from module.TrainerAutoEncoder import TrainerAutoEncoder
from module.TrainerBdt import TrainerBdt
from module.DataProcessor import DataProcessor

class Trainer:
    class ModelTypes(Enum):
        AutoEncoder = 0
        Bdt = 1

    def __init__(self, model_type, validation_data_fraction, test_data_fraction , **kwargs):
        self.model_type = model_type

        seed = np.random.randint(0, 99999999)

        data_processor = DataProcessor(validation_fraction=validation_data_fraction,
                                       test_fraction=test_data_fraction,
                                       seed=seed)
    
        self.model_trainer = None
        if model_type is self.ModelTypes.Bdt:
            self.model_trainer = TrainerBdt(data_processor=data_processor,
                                            seed=seed,
                                            validation_data_fraction=validation_data_fraction,
                                            test_data_fraction=test_data_fraction,
                                            **kwargs)
            
        elif model_type is self.ModelTypes.AutoEncoder:
            self.model_trainer = TrainerAutoEncoder(data_processor=data_processor,
                                                    seed=seed,
                                                    validation_data_fraction=validation_data_fraction,
                                                    test_data_fraction=test_data_fraction,
                                                    **kwargs)
        else:
            print("Unknown model type: ", model_type)
            
    
    def train(self, summaries_path):
        
        self.model_trainer.train()
        self.model_trainer.save_summary(path=summaries_path)