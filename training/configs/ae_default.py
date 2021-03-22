# ------------------------------------------------------------------------------------------------
# This is the default config file for the auto-encoder. Please copy it in case you want to change
# some parameters.
# ------------------------------------------------------------------------------------------------

# Model type
model_type = "AutoEncoder"
model_evaluator_path = "module/architectures/EvaluatorAutoEncoder.py"

# ---------------------------------------------
# Build general training settings dictionary

training_general_settings = {
    "model_trainer_path": "module/architectures/TrainerAutoEncoder.py",
    "validation_data_fraction": 0.15,
    "test_data_fraction": 0.15,
    "include_hlf": True,
    "include_efp": True,
    "hlf_to_drop": ['Energy', 'Flavor'],
}

# ---------------------------------------------
# Path to training data
# efp_base = 4
efp_base = 4
qcd_path = None
signals_base_path = None

if efp_base == 3:
    qcd_path = "../../data/backgrounds/qcd/h5_no_lepton_veto_fat_jets_dr0p8/*.h5"
    signals_base_path = "../../data/s_channel_delphes/h5_no_lepton_veto_fat_jets_dr0p8/"
elif efp_base == 4:
    qcd_path = "../../data/backgrounds/qcd/h5_no_lepton_veto_fat_jets_dr0p8_efp4/*.h5"
    signals_base_path = "../../data/s_channel_delphes/h5_no_lepton_veto_fat_jets_dr0p8_efp4/"
else:
    print("Invalid EFP base:", efp_base)
    qcd_path = None
    signals_base_path = None

# Path to testing data
input_path = signals_base_path+"/*/base_3/*.h5"

# ---------------------------------------------
# Output paths
# output_path = "trainingResults_previous_default/"
# output_path = "trainingResults_new_default/"
# output_path = "trainingResults_archs/"
# output_path = "trainingResults_new_config/"
output_path = "trainingResults_test/"

summary_path = output_path+"summary/"
results_path = output_path+"trainingRuns/"
AUCs_path = output_path+"aucs/"
stat_hists_path = output_path+"stat_hists.root"


# ---------------------------------------------
# Training parameters
training_params = {
    # 'batch_size': 32,
    'batch_size': 256, ##

    # losses documentation
    # https://keras.io/api/losses/regression_losses
    'loss': 'mean_absolute_error',
    # 'loss': 'mean_squared_error',
    # 'loss': 'mean_absolute_percentage_error',
    # 'loss': 'mean_squared_logarithmic_error', ##
    # 'loss': 'cosine_similarity',
    # 'loss': 'huber',
    # 'loss': 'log_cosh',

    # optimizers documentation
    # https://keras.io/api/optimizers/
    # 'optimizer': 'SGD',
    # 'optimizer': 'RMSprop',
    'optimizer': 'Adam',
    # 'optimizer': 'Adadelta',
    # 'optimizer': 'Adagrad',
    # 'optimizer': 'Adamax',
    # 'optimizer': 'Nadam',
    # 'optimizer': 'Ftrl',

    'metric': 'accuracy',

    # 'epochs': 200,
    'epochs': 2, ##
    
    'learning_rate': 0.00051,
    'es_patience': 12,
    'lr_patience': 9,
    'lr_factor': 0.5,
    
    # "bottleneck_size": 10, ##
    "bottleneck_size": 8,
    
    "intermediate_architecture": (30, 30),
    
}



# ---------------------------------------------
# Number of models to train
n_models = 1

# ---------------------------------------------
# Pick normalization type (definitions below):
norm_type = "StandardScaler"

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
# Once the training is done, you can specify
# which model was the best and use it for
# further tests/plotting
best_model = 0 ## new
# best_model = 9 ## old

# ---------------------------------------------
# Statistical analysis parameters
svj_jet_cut = 0.037
n_events_per_class = 10000

# ---------------------------------------------
# Output file names
# file_name = "hlf_eflow_{}_bottle_{}_default".format(efp_base, training_params["bottleneck_size"])

arch_summary = str(training_params["intermediate_architecture"]).replace("(","").replace(")","").replace(",","_").replace(" ","_")

# file_name = "hlf_eflow_{}_bottle_{}_arch_{}_loss_{}_batch_size_{}".format(efp_base,
#                                                                           training_params["bottleneck_size"],
#                                                                           arch_summary,
#                                                                           training_params["loss"],
#                                                                           training_params["batch_size"],
#                                                                           )

file_name = "hlf_bottle_{}_arch_{}_loss_{}_batch_size_{}".format(efp_base,
                                                                          training_params["bottleneck_size"],
                                                                          arch_summary,
                                                                          training_params["loss"],
                                                                          training_params["batch_size"],
                                                                          )


# ---------------------------------------------
# Build specific training settings dictionary (this will be passed to the specialized trainer class)
training_settings = {
    "qcd_path": qcd_path,
    "training_params": training_params,
    "EFP_base": efp_base,
    "norm_type": norm_type,
    "norm_args": normalizations[norm_type],
}
