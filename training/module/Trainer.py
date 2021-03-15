from enum import Enum

from module.TrainerAutoEncoder import TrainerAutoEncoder
from module.TrainerBdt import TrainerBdt

class Trainer:
    class ModelTypes(Enum):
        AutoEncoder = 0
        Bdt = 1

    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
    
        self.model_trainer = None
        if model_type is self.ModelTypes.Bdt:
            self.model_trainer = TrainerBdt(**kwargs)
        elif model_type is self.ModelTypes.AutoEncoder:
            self.model_trainer = TrainerAutoEncoder(**kwargs)
        else:
            print("Unknown model type: ", model_type)
            
    
    def train(self, summaries_path):
        
        self.model_trainer.train()
        self.model_trainer.save_summary(path=summaries_path)