import os
from pathlib import Path
from enum import Enum
import matplotlib.pyplot as plt
from collections import OrderedDict as odict
import numpy as np

import module.SummaryProcessor as summaryProcessor
import module.utils as utils
from module.DataProcessor import DataProcessor
from module.EvaluatorBdt import EvaluatorBdt
from module.EvaluatorAutoEncoder import EvaluatorAutoEncoder

plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams.update({'font.size': 18})

class Evaluator:
    
    class ModelTypes(Enum):
        AutoEncoder = 0
        Bdt = 1
    
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        
        self.model_evaluator = None
        if model_type is self.ModelTypes.Bdt:
            self.model_evaluator = EvaluatorBdt(**kwargs)
        elif model_type is self.ModelTypes.AutoEncoder:
            self.model_evaluator = EvaluatorAutoEncoder(**kwargs)
        else:
            print("Unknown model type: ", model_type)
    
    def save_aucs(self, summary_path, AUCs_path, **kwargs):
    
        summaries = summaryProcessor.get_summaries_from_path(summary_path)

        if not os.path.exists(AUCs_path):
            Path(AUCs_path).mkdir(parents=True, exist_ok=False)

        for _, summary in summaries.df.iterrows():
            
            filename = summary.training_output_path.split("/")[-1]
            utils.set_random_seed(summary.seed)
            data_processor = DataProcessor(summary=summary)
            
            auc_params = self.model_evaluator.get_aucs(summary=summary,
                                                       AUCs_path=AUCs_path,
                                                       filename=filename,
                                                       data_processor=data_processor,
                                                       **kwargs
                                                       )
            if auc_params is None:
                continue
            
            (aucs, auc_path, append, write_header) = auc_params
            self.__save_aucs_to_csv(aucs=aucs, path=auc_path, append=append, write_header=write_header)

    def __save_aucs_to_csv(self, aucs, path, append=False, write_header=True):
        # TODO: we could simplify what we store in the aucs file to m, r and auc only
    
        with open(path, "a" if append else "w") as out_file:
            if write_header:
                out_file.write(",name,auc,mass,nu\n")
        
            for index, dict in enumerate(aucs):
                out_file.write("{},Zprime_{}GeV_{},{},{},{}\n".format(index,
                                                                      dict["mass"], dict["rinv"], dict["auc"],
                                                                      dict["mass"], dict["rinv"]))

    def draw_roc_curves(self, summary_path, summary_version, **kwargs):

        summaries = summaryProcessor.get_summaries_from_path(summary_path)

        plotting_args = {k: v for k, v in kwargs.items() if k not in ["signals", "signals_base_path"] }

        fig, ax_begin, ax_end, plt_end, colors = self.__get_plot_params(n_plots=1, **plotting_args)
        ax = ax_begin(0)
        
        for _, summary in summaries.df.iterrows():
            version = summaryProcessor.get_version(summary.summary_path)
            if version != summary_version:
                continue

            utils.set_random_seed(summary.seed)
            kwargs["filename"] = summary.training_output_path.split("/")[-1]
            data_processor = DataProcessor(summary=summary)
        
            self.model_evaluator.draw_roc_curves(summary=summary,
                                                 data_processor=data_processor,
                                                 ax=ax,
                                                 colors=colors,
                                                 **kwargs
                                                 )

        x = [i for i in np.arange(0, 1.1, 0.1)]
        ax.plot(x, x, '--', c='black')
        ax_end("false positive rate", "true positive rate")
        plt_end()
        plt.show()

    def __get_plot_params(self,
                          n_plots,
                          cols=4,
                          figsize=20.,
                          yscale='linear',
                          xscale='linear',
                          figloc='lower right',
                          figname='Untitled',
                          savename=None,
                          ticksize=8,
                          fontsize=5,
                          colors=None
                          ):
        rows = n_plots / cols + bool(n_plots % cols)
        if n_plots < cols:
            cols = n_plots
            rows = 1
    
        if not isinstance(figsize, tuple):
            figsize = (figsize, rows * float(figsize) / cols)
    
        fig = plt.figure(figsize=figsize)
    
        def on_axis_begin(i):
            return plt.subplot(rows, cols, i + 1)
    
        def on_axis_end(xname, yname=''):
            plt.xlabel(xname + " ({0}-scaled)".format(xscale))
            plt.ylabel(yname + " ({0}-scaled)".format(yscale))
            plt.xticks(size=ticksize)
            plt.yticks(size=ticksize)
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.gca().spines['left']._adjust_location()
            plt.gca().spines['bottom']._adjust_location()
    
        def on_plot_end():
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = odict(list(zip(list(map(str, labels)), handles)))
            plt.figlegend(list(by_label.values()), list(by_label.keys()), loc=figloc)
            # plt.figlegend(handles, labels, loc=figloc)
            plt.suptitle(figname)
            plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01, rect=[0, 0.03, 1, 0.95])
            if savename is None:
                plt.show()
            else:
                plt.savefig(savename)
    
        if colors is None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if len(colors) < n_plots:
            print("too many plots for specified colors. overriding with RAINbow")
            import matplotlib.cm as cm
            colors = cm.rainbow(np.linspace(0, 1, n_plots))
        return fig, on_axis_begin, on_axis_end, on_plot_end, colors