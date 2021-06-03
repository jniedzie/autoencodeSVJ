# ------------------------------------------------------------------------------------------------
# This is the default config file for the auto-encoder. Please copy it in case you want to change
# some parameters.
# ------------------------------------------------------------------------------------------------

from ROOT import kBlue, kGreen, kRed, kOrange

train_on_signal = False

# ---------------------------------------------
# Output paths
output_path = "/Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training/trainingResults_pca/"

summary_path = output_path+"summary/"

results_path = output_path+"trainingRuns/"
plots_path = output_path+"plots/"
stat_hists_path = output_path+"stat_hists.root"

output_file_suffix = ""
# output_file_suffix = "_whiten"
# output_file_suffix = "_30constituents"

# ---------------------------------------------
# Build general training/evaluation settings dictionary

training_general_settings = {
    "model_trainer_path": "module/architectures/TrainerPca.py",
    "validation_data_fraction": 0.15,
    "test_data_fraction": 0.15,
    "include_hlf": True,
    "include_efp": True,
    "include_constituents": False,
    "hlf_to_drop": ['Energy', 'Flavor', "ChargedFraction"],
    "efp_to_drop": [str(i) for i in range(2, 13)],
    "constituents_to_drop": ["constituent_Rapidity_*", "constituent_Eta_*", "constituent_Phi_*"] + ["constituent_*_{}".format(i) for i in range(30, 150)],
    # "constituents_to_drop": []
    "max_jets": 2
}

evaluation_general_settings = {
    "model_evaluator_path": "module/architectures/EvaluatorPca.py",
    "summary_path": summary_path,
    "aucs_path": output_path+"aucs/",
}


# ---------------------------------------------
# Path to training data
efp_base = 3
# qcd_path = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_delphes/qcd/h5_no_lepton_veto_fat_jets_dr0p8_efp3_fatJetstrue_constituents150_maxJets2/base_{}/*.h5".format(efp_base)
qcd_path = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_delphes/qcd/h5_no_lepton_veto_fat_jets_dr0p8_efp3_fatJetstrue_constituents0_maxJets2/base_{}/*.h5".format(efp_base)


# signals_base_path = "/Users/Jeremi/Documents/Physics/ETH/data/s_channel_delphes/h5_no_lepton_veto_fat_jets_dr0p8_efp3_fatJetstrue_constituents150_maxJets2/"
signals_base_path = "/Users/Jeremi/Documents/Physics/ETH/data/s_channel_delphes/h5_no_lepton_veto_fat_jets_dr0p8_efp3_fatJetstrue_constituents0_maxJets2/"



# Path to testing data
input_path = signals_base_path+"/*/base_{}/*.h5".format(efp_base)


# ---------------------------------------------
# Training parameters
training_params = {
    # "n_components": 2,
    "n_components": "mle",
    "svd_solver": "full",
    "whiten": False,

}

# ---------------------------------------------
# Number of models to train
n_models = 5

# ---------------------------------------------
# Pick normalization type (definitions below):
norm_type = "MinMaxScaler"

# Set parameters for the selected normalization
normalizations = {
    # Custom implementation of robust scaler
    "Custom": {
        "norm_percentile": 25
    },
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
    # Custom implementation of the StandardScaler
    "CustomStandard": {
    },
    # Don't apply any scaling at all
    "None": {
    }
}

# ---------------------------------------------
# Once the training is done, you can specify which model was the best and use it for further tests/plotting
best_model = 6

# how many best models to include in the chi2 calculation
fraction_of_models_for_avg_chi2 = 0.8

# signal points for which tests will be done
# masses = [1500, 2000, 2500, 3000, 3500, 4000]
# masses = [2500, 3000, 3500, 4000]
test_masses = [3000]
# rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
test_rinvs = [0.75]

qcd_input_color = kGreen
qcd_reco_color = kBlue
signal_input_color = kRed
signal_reco_color = kOrange


# ---------------------------------------------
# Statistical analysis parameters
svj_jet_cut = 0.037
n_events_per_class = 10000

# ---------------------------------------------
# Output file names

file_name = "efp_{}".format(efp_base)
file_name += "_nComponents_{}".format(training_params["n_components"])
file_name += "_svdSolver_{}".format(training_params["svd_solver"])
file_name += "_{}".format(norm_type)
file_name += "_maxJets_{}".format(training_general_settings["max_jets"])
file_name += "{}".format(output_file_suffix)


# only files matching this pattern will be used for tests

test_filename_pattern = file_name+"_v"
# test_filename_pattern = "_v"

# ---------------------------------------------
# Build specific training/evaluation settings dictionary (this will be passed to the specialized trainer class)
training_settings = {
    "qcd_path": qcd_path,
    "training_params": training_params,
    "EFP_base": efp_base,
    "norm_type": norm_type,
    "norm_args": normalizations[norm_type],
}

evaluation_settings = {
    "input_path": input_path,
    "custom_objects": {}
}
