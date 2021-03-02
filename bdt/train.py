import sys
sys.path.append("../training")

from module.DataLoader import DataLoader

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

import pickle, os
import optparse
from pathlib import Path


parser = optparse.OptionParser("usage: %prog --method method")
parser.add_option("-i", "--i_sample", dest="i_sample", default=0, type=int)
(opt, args) = parser.parse_args()

i_sample = opt.i_sample

masses = [1500, 2000, 2500, 3000, 3500, 4000]
rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]

i_mass = i_sample % len(masses)
i_rinv = int(i_sample / len(masses))

# qcd_path = "../../data/training_data/qcd/base_3/*.h5"
# signal_path = "../../data/training_data/all_signals/{}GeV_{:3.2f}/base_3/*.h5".format(masses[i_mass], rinvs[i_rinv])
# model_output_path = "trainingResults/models/model_{}GeV_{:3.2f}.sav".format(masses[i_mass], rinvs[i_rinv])

# qcd_path = "../../data/backgrounds/qcd/h5_qcd/*.h5"
# qcd_path = "../../data/training_data/qcd/base_3/*.h5"
qcd_path = "../../data/backgrounds/qcd/h5_no_lepton_veto_fat_jets/*.h5"

# signal_path = "../../data/s_channel_delphes/h5_signal_no_MET_over_mt_cut/{}GeV_{:3.2f}/base_3/*.h5".format(masses[i_mass], rinvs[i_rinv])
signal_path = "../../data/s_channel_delphes/h5_no_lepton_veto_fat_jets/{}GeV_{:3.2f}/base_3/*.h5".format(masses[i_mass], rinvs[i_rinv])
model_output_path = "trainingResults_noLeptonVeto_fatJets/models/model_{}GeV_{:3.2f}.sav".format(masses[i_mass], rinvs[i_rinv])

print("Signal path: ", signal_path)
print("Model output path: ", model_output_path)


print("\n===================================")
print("Loading data")
print("===================================\n")

data_loader = DataLoader()
(X_train, Y_train), (X_test, Y_test) = data_loader.BDT_load_all_data(qcd_path=qcd_path, signal_path=signal_path)

print("\n===================================")
print("Fitting a model")
print("===================================\n")

bdt = AdaBoostClassifier(algorithm='SAMME', n_estimators=800, learning_rate=0.5)
bdt.fit(X_train, Y_train)

print("\n===================================")
print("Fit done. Saving model")
print("===================================\n")

Path(os.path.dirname(model_output_path)).mkdir(parents=True, exist_ok=True)
pickle.dump(bdt, open(model_output_path, 'wb'))

Y_predicted = bdt.predict(X_test)
report = classification_report(Y_test, Y_predicted, target_names=["background", "signal"])
print("Report: ", report)

