import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import importlib, argparse
import csv

import sys
sys.path.insert(1, "../")


# ------------------------------------------------------------------------------------------------
# This script will draw a table with AUCs (areas under ROC curve) values based on the CSV
# file stored in "AUCs_path" with version specified by "training_version" from the provided
# config file.
# ------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("-c", "--config", dest="config_path", default=None, required=True, help="Path to the config file")
args = parser.parse_args()
config_path = args.config_path.replace(".py", "").replace("../", "").replace("/", ".")
config = importlib.import_module(config_path)

n_bins_rinv = 3
n_bins_mass = 6

matplotlib.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def plot_aucs(dataframe, title=None):
    fac = 1.5
    
    plt.figure(figsize=(1.1 * fac * 6.9, 1.1 * fac * 6))
    plt.imshow(dataframe, cmap='viridis')
    
    cb = plt.colorbar()
    cb.set_label(label='AUC value', fontsize=18 * fac)
    plt.clim(0.58, 0.9)
    
    plt.xticks(np.arange(0, n_bins_rinv, 1), map(lambda x: '{:.2f}'.format(float(x)), np.unique(dataframe.columns)))
    plt.yticks(np.arange(0, n_bins_mass, 1), np.unique(dataframe.index))
    
    plt.title(title, fontsize=fac * 25)
    plt.ylabel(r'$M_{Z^\prime}$ (GeV)', fontsize=fac * 20)
    plt.xlabel(r'$r_{inv}$', fontsize=fac * 20)
    plt.xticks(fontsize=18 * fac)
    plt.yticks(fontsize=18 * fac)
    
    for mi, (mass, row) in enumerate(dataframe.iterrows()):
        for ni, (nu, auc) in enumerate(row.iteritems()):
            plt.text(ni, mi, '{:.3f}'.format(auc), ha="center", va="center", color="w", fontsize=18 * fac)


def read_csv_file():
    aucs_path = config.evaluation_general_settings["aucs_path"]
    filename = aucs_path + config.file_name + "_v" + str(config.best_model)
    data = {}
    masses = []
    rinvs = []
    
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        
        first = True
        
        for row in reader:
            if first:
                first = False
                continue
            
            mass = row[0]
            rinv = row[1]
            auc = row[2]
            
            data[(mass, rinv)] = auc
            
            if mass not in masses:
                masses.append(mass)
                
            if rinv not in rinvs:
                rinvs.append(rinv)
    
    masses.sort()
    rinvs.sort()
    
    return masses, rinvs, data


def produce_dataframe(masses, rinvs, data):
    columns = {}
    
    for mass in masses:
        for rinv in rinvs:
            if rinv not in columns.keys():
                columns[rinv] = []

            if (mass, rinv) in data.keys():
                columns[rinv].append(data[(mass, rinv)])
            else:
                columns[rinv].append(0.0)
    
    return pd.DataFrame(columns, index=masses, dtype=float)


masses, rinvs, data = read_csv_file()
dataframe = produce_dataframe(masses, rinvs, data)

plot_aucs(dataframe)
plt.show()