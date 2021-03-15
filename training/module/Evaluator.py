import os
from pathlib import Path
from enum import Enum

import module.SummaryProcessor as summaryProcessor
import module.utils as utils
from module.DataProcessor import DataProcessor
from module.EvaluatorBdt import EvaluatorBdt
from module.EvaluatorAutoEncoder import EvaluatorAutoEncoder


class Evaluator:
    
    class ModelTypes(Enum):
        AutoEncoder = 0
        Bdt = 1
    
    def __init__(self, model_type):
        self.model_type = model_type
        
        self.model_evaluator = None
        if model_type is self.ModelTypes.Bdt:
            self.model_evaluator = EvaluatorBdt()
        elif model_type is self.ModelTypes.AutoEncoder:
            self.model_evaluator = EvaluatorAutoEncoder()
        else:
            print("Unknown model type: ", model_type)
    
    def save_aucs(self, summary_path, AUCs_path, **kwargs):
    
        summaries = summaryProcessor.summary(summary_path=summary_path)

        if not os.path.exists(AUCs_path):
            Path(AUCs_path).mkdir(parents=True, exist_ok=False)

        for index, summary in summaries.df.iterrows():
            utils.set_random_seed(summary.seed)
            filename = summary.training_output_path.split("/")[-1]
    
            data_processor = DataProcessor(validation_fraction=summary.val_split,
                                           test_fraction=summary.test_split,
                                           seed=summary.seed
                                           )
                
            self.model_evaluator.save_aucs(summary=summary,
                                           AUCs_path=AUCs_path,
                                           filename=filename,
                                           data_processor=data_processor,
                                           **kwargs
                                           )
