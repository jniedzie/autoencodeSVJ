# ------------------------------------------------------------------------------------------------
# This is the default config file for the auto-encoder. Please copy it in case you want to change
# some parameters.
# ------------------------------------------------------------------------------------------------


# ---------------------------------------------
# Input data path
# qcd_path = "../../data/backgrounds/qcd/h5_qcd/*.h5"
qcd_path = "../../data/training_data/qcd/base_3/*.h5"

# ---------------------------------------------
# Output paths
output_path = "trainingResults_test/"
summary_path = output_path+"summary/"
results_path = output_path+"trainingRuns/"

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

target_dim = 8

# ---------------------------------------------
# Number of models to train
n_models = 5

# ---------------------------------------------
# Pick normalization type (definitions below):
norm_name = "StandardScaler"

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

norm_args = normalizations[norm_name]
