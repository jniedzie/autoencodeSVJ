# ------------------------------------------------------------------------------------------------
# This is the default config file for the auto-encoder. Please copy it in case you want to change
# some parameters.
# ------------------------------------------------------------------------------------------------


# Model type
model_type = "BDT"
model_evaluator_path = "module/architectures/EvaluatorBdt.py"

training_general_settings = {
    "model_trainer_path": "module/architectures/TrainerBdt.py",
    "validation_data_fraction": 0.0,
    "test_data_fraction": 0.2,
    "include_hlf": True,
    "include_efp": True,
    "hlf_to_drop": ['Energy', 'Flavor'],
}

efp_base = 4

# ---------------------------------------------
# Path to training data
qcd_path = "../../data/backgrounds/qcd/h5_no_lepton_veto_fat_jets_dr0p8/base_{}/*.h5".format(efp_base)

# Path to testing data
signals_base_path = "../../data/s_channel_delphes/h5_no_lepton_veto_fat_jets_dr0p8/"

# ---------------------------------------------
# Output paths
output_path = "trainingResults_test/bdt/"
summary_path = output_path+"summary/"
results_path = output_path+"trainingRuns/"
AUCs_path = output_path+"aucs/"
stat_hists_path = output_path+"stat_hists.root"

# ---------------------------------------------
# Training parameters
training_params = {
    'batch_size': 32,
    'loss': 'mse',
    'optimizer': 'adam',
    'epochs': 200,
    'learning_rate': 0.00051,
    'es_patience': 12,
    'lr_patience': 9,
    'lr_factor': 0.5
}



# ---------------------------------------------
# Number of models to train
n_models = 2

# ---------------------------------------------
# Pick normalization type (definitions below):
norm_type = "None"

# Set parameters for the selected normalization
normalizations = {
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
    "RobustScaler": {
        "quantile_range": (0.25, 0.75),
        "with_centering": True,
        "with_scaling": True,
        "copy": True,
    },
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
    "MinMaxScaler": {
        "feature_range": (0, 1),
        "copy": True,
    },
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
    "StandardScaler": {
        "with_mean": True,
        "copy": True,
        "with_std": True,
    },
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler
    "MaxAbsScaler": {
        "copy": True,
    },
    # Don't apply any scaling at all
    "None": {}
}

# ---------------------------------------------
# Once the training is done, you can specify
# which model was the best and use it for
# further tests/plotting
best_model = 0

# ---------------------------------------------
# Statistical analysis parameters
svj_jet_cut = 0.037
n_events_per_class = 10000

# ---------------------------------------------
# Output file names
file_name = "hlf_eflow_{}".format(efp_base)

# ---------------------------------------------
# Build specific training settings dictionary (this will be passed to the specialized trainer class)
training_settings = {
    "qcd_path": qcd_path,
    "training_params": training_params,
    "EFP_base": efp_base,
    "norm_type": norm_type,
    "norm_args": normalizations[norm_type],
}