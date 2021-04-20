# ------------------------------------------------------------------------------------------------
# This is the default config file for the auto-encoder. Please copy it in case you want to change
# some parameters.
# ------------------------------------------------------------------------------------------------

from ROOT import kBlue, kGreen, kRed, kOrange
from module.architectures.DenseTiedLayer import DenseTiedLayer

# Model type
model_type = "AutoEncoder"

train_on_signal = False


# ---------------------------------------------
# Output paths
# output_path = "/Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training/trainingResults_excluding_eta_phi/"
# output_path = "/Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training/trainingResults_activations/"
# output_path = "/Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training/trainingResults_small_archs/"
# output_path = "/Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training/trainingResults_norms/"
# output_path = "/Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training/trainingResults_1_efp/"
# output_path = "/Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training/trainingResults_archs/"
# output_path = "/Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training/trainingResults_tiedWeights/"
# output_path = "/Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training/trainingResults_flatAe/"

output_path = "/Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training/trainingResults_oldSamples/"

summary_path = output_path+"summary/"

results_path = output_path+"trainingRuns/"
plots_path = output_path+"plots/"
stat_hists_path = output_path+"stat_hists.root"

output_file_suffix = ""

# ---------------------------------------------
# Build general training/evaluation settings dictionary

training_general_settings = {
    "model_trainer_path": "module/architectures/TrainerAutoEncoder.py",
    "validation_data_fraction": 0.15,
    "test_data_fraction": 0.15,
    "include_hlf": True,
    "include_efp": True,
    "hlf_to_drop": ['Energy', 'Flavor', "ChargedFraction"],
    "efp_to_drop": [str(i) for i in range(2, 13)],
}

evaluation_general_settings = {
    "model_evaluator_path": "module/architectures/EvaluatorAutoEncoder.py",
    "summary_path": summary_path,
    "aucs_path": output_path+"aucs/",
}


# ---------------------------------------------
# Path to training data
efp_base = 3
# qcd_path = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds/qcd/h5_no_lepton_veto_fat_jets_dr0p8/base_{}/*.h5".format(efp_base)
qcd_path = "/Users/Jeremi/Documents/Physics/ETH/data/old_training_data/qcd/base_{}/*.h5".format(efp_base)

# signals_base_path = "/Users/Jeremi/Documents/Physics/ETH/data/s_channel_delphes/h5_no_lepton_veto_fat_jets_dr0p8/"
signals_base_path = "/Users/Jeremi/Documents/Physics/ETH/data/old_training_data/all_signals/"

# Path to testing data
input_path = signals_base_path+"/*/base_{}/*.h5".format(efp_base)


# ---------------------------------------------
# Training parameters
training_params = {
    'batch_size': 256,

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

    # metrics documentation
    # https://keras.io/api/metrics/
    'metric': 'accuracy',
    # 'metric': 'binary_accuracy',
    # 'metric': 'categorical_accuracy',
    # 'metric': 'top_k_categorical_accuracy',
    # 'metric': 'sparse_top_k_categorical_accuracy',

    # 'metric': 'binary_crossentropy',
    # 'metric': 'categorical_crossentropy',
    # 'metric': 'sparse_categorical_crossentropy',
    # 'metric': 'kullback_leibler_divergence',
    # 'metric': 'poisson',

    # 'metric': 'mean_squared_error',
    # 'metric': 'root_mean_squared_error',
    # 'metric': 'mean_absolute_error',
    # 'metric': 'mean_absolute_percentage_error',
    # 'metric': 'mean_squared_logarithmic_error',
    # 'metric': 'cosine_similarity',
    # 'metric': 'logcosh',

    
    'epochs': 200,
    
    'learning_rate': 0.00051,
    'es_patience': 12,
    'lr_patience': 9,
    'lr_factor': 0.5,
    
    "bottleneck_size": 5,
    
    "intermediate_architecture": (7, ),

    # activation functions documentation
    # https://keras.io/api/layers/activations/
    # "activation": "relu",
    "activation": "elu",
    # "activation": "sigmoid",
    # "activation": "softmax",
    # "activation": "softplus",
    # "activation": "softsign",
    # "activation": "tanh",
    # "activation": "selu",
    # "activation": "exponential",
    
    "output_activation": "linear",

    "tied_weights": False
}

# ---------------------------------------------
# Number of models to train
n_models = 5

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
# Once the training is done, you can specify which model was the best and use it for further tests/plotting
best_model = 0

# how many best models to include in the chi2 calculation
fraction_of_models_for_avg_chi2 = 0.8

# only files matching this pattern will be used for tests




# signal points for which tests will be done
# masses = [1500, 2000, 2500, 3000, 3500, 4000]
# masses = [2500, 3000, 3500, 4000]
test_masses = [3000]
# rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
test_rinvs = [0.45]

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
arch_summary = str(training_params["intermediate_architecture"]).replace("(","").replace(")","").replace(",","_").replace(" ","_")

file_name = "efp_{}".format(efp_base)
file_name += "_bottle_{}".format(training_params["bottleneck_size"])
file_name += "_arch_{}".format(arch_summary)
file_name += "_loss_{}".format(training_params["loss"])
file_name += "_optimizer_{}".format(training_params["optimizer"])
file_name += "_batchSize_{}".format(training_params["batch_size"])
file_name += "_{}".format(norm_type)
file_name += "_activation_{}".format(training_params["activation"])
file_name += "_tiedWeights_{}".format(training_params["tied_weights"])
file_name += "_epochs_{}".format(training_params["epochs"])
file_name += "{}".format(output_file_suffix)

test_filename_pattern = file_name+"_v"
# test_filename_pattern = "hlf_efp_3_bottle_5_arch_7__7_loss_mean_absolute_error_optimizer_Adam_batch_size_256_scaler_StandardScaler_activation_elu_same_train_val_scaler_v"
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
    "custom_objects": {
        "DenseTiedLayer": DenseTiedLayer,
    }
}
