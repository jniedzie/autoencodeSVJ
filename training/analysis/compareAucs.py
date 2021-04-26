import csv
from ROOT import TLegend, TH1D, TCanvas, gApplication, TGraph, TGraphErrors
from ROOT import kRed, kOrange, kGreen, kBlue, kViolet, kBlack
import glob

show_plots = False

input_paths = [

    (glob.glob("../trainingResults_withConstituents/aucs/efp_3_bottle_50_arch_20__20_loss_mean_absolute_error_optimizer_Adam_batchSize_256_StandardScaler_activation_elu_tiedWeights_False_epochs_200_23constituents_v*"), kBlue, "(7, 7), BN: 1"),
    (glob.glob("../trainingResults_withConstituents/aucs/efp_3_bottle_50_arch_50__50_loss_mean_absolute_error_optimizer_Adam_batchSize_256_StandardScaler_activation_elu_tiedWeights_False_epochs_200_23constituents_v*"), kBlue, "(7, 7), BN: 1"),
    (glob.glob("../trainingResults_withConstituents/aucs/efp_3_bottle_10_arch_20__20_loss_mean_absolute_error_optimizer_Adam_batchSize_256_StandardScaler_activation_elu_tiedWeights_False_epochs_200_23constituents_v*"), kBlue, "(7, 7), BN: 1"),
]

draw_auc_distributions = False

r_invs = [0.15, 0.30, 0.45, 0.60, 0.75]
masses = [1500, 2000, 2500, 3000, 3500, 4000]
colors = [kRed, kOrange, kGreen, kGreen+2, kBlue, kViolet, kBlack]

xMin = 0.0
xMax = 1.0

nBinsRinv = 130
nBinsMass = 50
plotsTitle = ""

min_max = {
    "mean": (0.45, 0.7),
    "sd": (0.0, 0.08),
    "best": (0.55, 0.8),
}


def getHistogramForVariable(stats, variableValue, forMass):

    title = "mass" if forMass else "r_inv"
    title += " hist" + str(variableValue)
    hist = TH1D(title, title,  nBinsMass if forMass else nBinsRinv, xMin, xMax)

    for stat in stats:

        for result in stat:
            if (result[0] if forMass else result[1]) == variableValue:
                hist.Fill(result[2])
    return hist


def drawHistsForVariable(stats, forMass):
    
    leg = TLegend(0.1, 0.6, 0.5, 0.9)

    var_name = "mass" if forMass else "r_inv"
    print("AUCs per ", var_name, ": ")
    print("mean\tmeanErr\twidth\twidthErr\tmax\tmaxErr")

    hists = []

    for i in range(len(masses) if forMass else len(r_invs)):
    
        hist = getHistogramForVariable(stats,  masses[i] if forMass else r_invs[i], forMass)
        hist.Sumw2()
        hist.SetLineColor(colors[i])
        
        if i == 0:
            hist.SetTitle(plotsTitle)
            hist.GetXaxis().SetTitle("AUC")
            hist.GetYaxis().SetTitle("# trainings")

        value = masses[i] if forMass else r_invs[i]
        hists.append({"value": value, "hist": hist})

        if forMass:
            title = "m = " + str(masses[i]) + " GeV"
        else:
            title = "r_{inv} = " + str(r_invs[i])
        leg.AddEntry(hist, title, "l")

    return hists, leg
    
def print_average_auc_for_files(paths):
    
    auc_per_path = []

    for path in paths:
        average_auc = 0
        count = 0
        
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            first_row = True
            
            for row in spamreader:
                if first_row:
                    first_row = False
                    continue
                
                values = row[0].split(",")
                
                average_auc += float(values[2])
                count += 1

        average_auc /= count
        
        auc_per_path.append((average_auc, path))
    
    auc_per_path = sorted(auc_per_path, key=lambda x: x[0])

    avg_auc = 0

    for auc, path in auc_per_path:
        avg_auc += auc

    print("Avg auc for file: ", paths[0], "is: ", avg_auc/len(auc_per_path))
        

def get_auc_params(csv_file_paths, forMass):
    model_stats = []

    for csv_file_path in csv_file_paths:
    
        with open(csv_file_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            first_row = True
    
            for row in spamreader:
                if first_row:
                    first_row = False
                    continue
    
                values = row[0].split(",")

                mass = float(values[0])
                rinv = float(values[1])
                auc = float(values[2])

    
                model_stats.append((mass, rinv, auc))
    
                # print(mass, "\t", rinv, "\t", auc)

    hists, leg = drawHistsForVariable([model_stats], forMass)

    if draw_auc_distributions:
        canvas = TCanvas("", "", 800, 600)
        canvas.cd()
        hists[0]["hist"].Draw()

        for values_hist in hists:
            values_hist["hist"].Draw("same")
        
        leg.Draw()
        canvas.Update()
        # gApplication.Run()
        
        
    auc_params = {}
    
    for values_hist in hists:
        value = values_hist["value"]
        hist = values_hist["hist"]
        
        hist.Draw("same")

        mean = hist.GetMean()
        meanErr = hist.GetMeanError()
        sd = hist.GetStdDev()
        sdErr = hist.GetStdDevError()
        best = hist.GetXaxis().GetBinCenter(hist.FindLastBinAbove(0))
        bestErr = hist.GetXaxis().GetBinWidth(0) / 2.
        
        auc_params[value] = {
            "mean": mean, "meanErr": meanErr,
            "sd": sd, "sdErr": sdErr,
            "best": best, "bestErr": bestErr
        }
        
    return auc_params

def get_graphs_from_path(path, forMass):
    auc_params = get_auc_params(path, forMass=forMass)
    
    graphs = {
        "mean": TGraphErrors(),
        "sd": TGraphErrors(),
        "best": TGraphErrors(),
    }
    
    i_point = 0
    for value, params in auc_params.items():
        for name, graph in graphs.items():
            graph.SetPoint(i_point, value, params[name])
            graph.SetPointError(i_point, 0, params[name + "Err"])
        
        i_point += 1
    
    return graphs
    
    
if __name__ == "__main__":
    
    canvas = TCanvas("", "", 800, 600)
    canvas.Divide(3, 2)
    
    first_graph = True
    
    graphs_per_path_per_rinv = {}
    graphs_per_path_per_mass = {}
    
    for path, _, _ in input_paths:
        if len(path) == 0:
            print("Can't load AUC's, maybe you forgot to run produceMissingAucs.py ?")
            continue

        graphs_per_path_per_rinv[path[0]] = get_graphs_from_path(path, forMass=True)
        graphs_per_path_per_mass[path[0]] = get_graphs_from_path(path, forMass=False)
        
        print_average_auc_for_files(path)

    if not show_plots:
        exit()

    legend = TLegend(0.5, 0.7, 0.9, 0.9)
    
    for path, color, title in input_paths:
    
        graphs_per_mass = graphs_per_path_per_mass[path[0]]
        graphs_per_rinv = graphs_per_path_per_rinv[path[0]]
        
        i_pad = 1
        for name, graph in graphs_per_mass.items():
            canvas.cd(i_pad)
            graph.SetMarkerStyle(20)
            graph.SetMarkerSize(1)
            graph.SetMarkerColor(color)
            
            if first_graph:
                graph.Draw("APE")
                graph.GetXaxis().SetTitle("r_{inv}")
                graph.SetMinimum(min_max[name][0])
                graph.SetMaximum(min_max[name][1])
            else:
                graph.Draw("PEsame")

            if i_pad == 1:
                legend.AddEntry(graph, title, "P")
                
            i_pad += 1


        for name, graph in graphs_per_rinv.items():
            canvas.cd(i_pad)
            graph.SetMarkerStyle(20)
            graph.SetMarkerSize(1)
            graph.SetMarkerColor(color)
    
            if first_graph:
                graph.Draw("APE")
                graph.GetXaxis().SetTitle("m_{Z'} (GeV)")
                graph.SetMinimum(min_max[name][0])
                graph.SetMaximum(min_max[name][1])
            else:
                graph.Draw("PEsame")
            i_pad += 1
        
        first_graph = False


    canvas.cd(1)
    legend.Draw()

    canvas.Update()
    
    gApplication.Run()
        
        
        
        
    
