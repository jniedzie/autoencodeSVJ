import module.SummaryProcessor as summaryProcessor
from module.AucGetter import AucGetter
from module.DataHolder import DataHolder
import importlib, argparse
import glob, os
from pathlib import Path
import tensorflow as tf

# ------------------------------------------------------------------------------------------------
# This script will produce a CSV file with areas under ROC curves (AUCs) for each training
# summary file found in the "summary_path", testing on all signal samples found
# in "input_path" and store the output in "aucs_path", as defined in provided config file.
# ------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("-c", "--config", dest="config_path", default=None, required=True, help="Path to the config file")
args = parser.parse_args()
config = importlib.import_module(args.config_path)

signalDict = {}
for path in glob.glob(config.input_path):
    key = path.split("/")[-3]
    signalDict[key] = path

summaries = summaryProcessor.summary(summary_path=config.summary_path)

if not os.path.exists(config.AUCs_path):
    Path(config.AUCs_path).mkdir(parents=True, exist_ok=False)

for index, row in summaries.df.iterrows():
    path = row.training_output_path
    filename = path.split("/")[-1]
    auc_path = config.AUCs_path + "/" + filename
    
    if not os.path.exists(auc_path):
        tf.compat.v1.reset_default_graph()
        
        auc_getter = AucGetter(filename=filename, summary_path=config.summary_path)
        
        data_holder = DataHolder(qcd=row.qcd_path, **signalDict)
        data_holder.load()
        
        norm, err, recon = auc_getter.get_errs_recon(data_holder)
        
        ROCs = auc_getter.get_aucs(err)
        AUCs = auc_getter.auc_metric(ROCs)
        AUCs.to_csv(auc_path)