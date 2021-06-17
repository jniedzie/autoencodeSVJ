from ROOT import TH1D, TH2D, kGreen, kBlue, TCanvas, gApplication, gStyle, TLegend, kRed, gPad, kOrange

bins = {
    "Eta": (-3.5, 3.5, 100),
    "Rapidity": (-3.5, 3.5, 100),
    "Phi": (-3.5, 3.5, 100),
    "Pt": (0, 2000, 100),

    "M": (0, 800, 100),
    "constituent_Energy": (0, 800, 100),
    "constituent_PT": (0, 2000, 100),
    "ChargedFraction": (0, 1, 100),
    "PTD": (0, 1, 100),
    "Axis2": (0, 0.2, 100),
    "efp": (0, 1, 100),
    "loss": (0.0, 300, 50),
    "latent": (-1, 4, 150),
}


def get_histogram(data, variable_name, color=None, suffix="", weights=None):
    """
    Returns TH1D filled with data for the specified variable.
    """
    
    hist = __initialize_histogram(variable_name, suffix)
    __fill_histogram(hist, data, variable_name, weights)
    if color is not None:
        hist.SetLineColor(color)
    return hist


def get_histogram_2d(data, variable_name_x, variable_name_y, suffix=""):
    """
    Returns TH2D filled with data for specified variables x and y.
    """

    hist = __initialize_histogram_2d(variable_name_x, variable_name_y, suffix)
    __fill_histogram_2d(hist, data, variable_name_x, variable_name_y)
    return hist


def get_hists_chi_2(hist_1, hist_2):
    """
    Returns chi2/NDF difference between two histograms
    """
    
    return hist_1.Chi2Test(hist_2, "CHI2/NDF")


def get_signal_paths(config):
    signal_paths = []
    
    for mass in config.test_masses:
        for rinv in config.test_rinvs:
            path = "{}{}GeV_{:1.2f}_mDark20_alphaPeak/base_{}/*.h5".format(config.signals_base_path, mass, rinv, config.efp_base)
            signal_paths.append(path)
    
    return signal_paths


def __get_binning_for_variable(variable_name):
    """
    Returns (n_bins, min, max) for a histogram, based on the variable name
    """

    special_keys = ["efp", "latent", "Eta", "Phi", "constituent_PT", "constituent_Energy", "Rapidity"]

    for key in special_keys:
        if key in variable_name:
            variable_name = key

    min = bins[variable_name][0]
    max = bins[variable_name][1]
    n_bins = bins[variable_name][2]
    
    return n_bins, min, max


def __initialize_histogram(variable_name, suffix=""):
    """
    Returns TH1D with correct binning and title for given variable.
    """
    
    n_bins, min, max = __get_binning_for_variable(variable_name)
    return TH1D(variable_name + suffix, variable_name + suffix, n_bins, min, max)


def __initialize_histogram_2d(variable_name_x, variable_name_y, suffix=""):
    """
    Returns TH1D with correct binning and title for given variable.
    """

    n_bins_x, min_x, max_x = __get_binning_for_variable(variable_name_x)
    n_bins_y, min_y, max_y = __get_binning_for_variable(variable_name_y)
    title = variable_name_x + variable_name_y + suffix
    return TH2D(title, title, n_bins_x, min_x, max_x, n_bins_y, min_y, max_y)

def __fill_histogram(histogram, data, variable_name, weights=None):
    """
    Fills provided histogram with data.
    """
    if variable_name == "loss":
        values = data
    elif "latent" in variable_name:
        values = data[int(variable_name.replace("latent_", ""))].tolist()
    else:
        values = data[variable_name].tolist()
        
    if weights is not None:
        for value, weight in zip(values, weights):
            histogram.Fill(value, weight)
    else:
        for value in values:
            histogram.Fill(value)

def __fill_histogram_2d(histogram, data, variable_name_x, variable_name_y):
    """
    Fills provided histogram with data.
    """

    if "latent" in variable_name_x:
        variable_name_x = int(variable_name_x.replace("latent_", ""))

    if "latent" in variable_name_y:
        variable_name_y = int(variable_name_y.replace("latent_", ""))

    values_x = data[variable_name_x].tolist()
    values_y = data[variable_name_y].tolist()

    for x, y in zip(values_x, values_y):
        histogram.Fill(x, y)
