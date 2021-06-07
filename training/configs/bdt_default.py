# ------------------------------------------------------------------------------------------------
# This is the default config file for the BDT. Please copy it in case you want to change
# some parameters.
# ------------------------------------------------------------------------------------------------

# ---------------------------------------------
# Model type (used by training/evaluation scripts to determine input/output file names)
model_type = "BDT"

train_on_signal = True

# ---------------------------------------------
# Output paths
output_path = "/Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training/trainingResults_bdt/"
summary_path = output_path+"summary/"

results_path = output_path+"trainingRuns/"
plots_path = output_path+"plots/"
stat_hists_path = output_path+"stat_hists.root"

# output_file_suffix = "_noConstituents_onlyEtaPhiEfp"
# output_file_suffix = "_noConstituents_onlyEtaPhiAxis2"
# output_file_suffix = "_noConstituents_onlyEtaPhiPtd"
# output_file_suffix = "_noConstituents_onlyEtaPhiPt"
# output_file_suffix = "_noConstituents_onlyEtaPhiMass"
# output_file_suffix = "_noConstituents_oneEFP_balancedSignal"
# output_file_suffix = "_noConstituents_oneEFP"
# output_file_suffix = "_noConstituents_oneEFP_oneJet"
# output_file_suffix = "_noConstituents_oneEFP_threeJets"
# output_file_suffix = "_30Constituents_noEFP"
output_file_suffix = "_100Constituents_noEFP"

# ---------------------------------------------
# General training/evaluation settings
training_general_settings = {
    "model_trainer_path": "module/architectures/TrainerBdt.py",
    "validation_data_fraction": 0.0,
    "test_data_fraction": 0.2,
    "include_hlf": True,
    "include_efp": False,
    "include_constituents": False,
    # "hlf_to_drop": ["Energy", "Flavor", "ChargedFraction", "M", "Pt", "PTD", "Axis2"],
    "hlf_to_drop": ["Energy", "Flavor", "ChargedFraction", "M", "Pt", "PTD"],
    # "hlf_to_drop": ["Energy", "Flavor", "ChargedFraction"],
    "efp_to_drop": [str(i) for i in range(2, 13)],
    "constituents_to_drop": ["constituent_Rapidity_*", "constituent_Eta_*", "constituent_Phi_*"] + ["constituent_*_{}".format(i) for i in range(100, 150)],
    "max_jets": 2
}

evaluation_general_settings = {
    "model_evaluator_path": "module/architectures/EvaluatorBdt.py",
    "summary_path": summary_path,
    "aucs_path": output_path+"aucs/",
}

# ---------------------------------------------
# Number of models to train
n_models = 1

# ---------------------------------------------
# Input data paths
efp_base = 3
qcd_path = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_delphes/qcd/h5_no_lepton_veto_fat_jets_dr0p8_efp3_fatJetstrue_constituents150_maxJets2/base_{}/*.h5".format(efp_base)
signals_base_path = "/Users/Jeremi/Documents/Physics/ETH/data/s_channel_delphes/h5_no_lepton_veto_fat_jets_dr0p8_efp3_fatJetstrue_constituents150_maxJets2/"


# ---------------------------------------------
# Training parameters
training_params = {
    "algorithm": "SAMME",
    "n_estimators": 800,
    "learning_rate": 0.5,
}

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
# Output file name
file_name = "efp_{}".format(efp_base)
file_name += "_norm_{}".format(norm_type)
file_name += "_algo_{}".format(training_params["algorithm"])
file_name += "_nEstimators_{}".format(training_params["n_estimators"])
file_name += "_learningRate_{}".format(training_params["learning_rate"])
file_name += "{}".format(output_file_suffix)

test_filename_pattern = file_name

# ---------------------------------------------
# Architecture specific training settings (this will be passed to the specialized trainer class)
training_settings = {
    "qcd_path": qcd_path,
    "training_params": training_params,
    "EFP_base": efp_base,
    "norm_type": norm_type,
    "norm_args": normalizations[norm_type],
}

evaluation_settings = {
    "input_path": signals_base_path,
}

# ---------------------------------------------
# Once the training is done, you can specify which model was the best and use it for further tests/plotting
best_model = 0

# ---------------------------------------------
# Statistical analysis parameters
svj_jet_cut = 0.037
n_events_per_class = 10000
