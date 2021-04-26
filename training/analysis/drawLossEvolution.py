
from ROOT import TGraph, kRed, kGreen, TLegend, TCanvas, gStyle
import csv, glob

aucsPath = "../trainingResults_archs/aucs/"
resultsPath = "../trainingResults_archs/trainingRuns/"
filePattern = "hlf_efp_3_bottle_9_arch_60__60_loss_mean_absolute_error_batch_size_256_v"


input_path = glob.glob("../trainingResults_withConstituents/trainingRuns/efp_3_bottle_5_arch_5__5_loss_mean_absolute_error_optimizer_Adam_batchSize_256_StandardScaler_activation_elu_tiedWeights_False_epochs_200_23constituents_v*.csv")

saved_plots = []


def draw_graphs(stats):

    canvas = TCanvas("c1", "c1", 2880, 1800)
    canvas.Divide(5, 2)

    legend = TLegend(0.5, 0.7, 0.9, 0.9)

    training_loss = []
    validation_loss = []

    for i_pad, stats_for_file in enumerate(stats):

        canvas.cd(i_pad+1)

        training_loss.append(TGraph())
        validation_loss.append(TGraph())

        for i, stat in enumerate(stats_for_file):
            training_loss[-1].SetPoint(i, i, stat[2])
            validation_loss[-1].SetPoint(i, i, stat[5])

        training_loss[-1].SetMarkerStyle(20)
        training_loss[-1].SetMarkerSize(1.0)
        training_loss[-1].SetMarkerColor(kRed + 1)
        training_loss[-1].SetTitle("v"+str(i_pad))

        validation_loss[-1].SetMarkerStyle(20)
        validation_loss[-1].SetMarkerSize(1.0)
        validation_loss[-1].SetMarkerColor(kGreen + 1)

        training_loss[-1].Draw("AP")
        validation_loss[-1].Draw("Psame")

        training_loss[-1].GetXaxis().SetTitle("Epoch")
        training_loss[-1].GetYaxis().SetTitle("Loss")

        training_loss[-1].GetXaxis().SetLimits(0, 200)

        if i_pad == 0:
            legend.AddEntry(training_loss[-1], "training loss", "p")
            legend.AddEntry(validation_loss[-1], "validation loss", "p")

    legend.Draw()
    canvas.Update()

    filename = input_path[0]
    filename = filename.replace("trainingRuns", "plots")
    filename = filename.replace(".csv", ".pdf")
    canvas.SaveAs(filename)
    saved_plots.append(filename)


def get_stats_from_file(inFilePath):

    stats = []

    with open(inFilePath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        first_row = True

        for row in spamreader:
            if first_row:
                first_row = False
                continue

            values = row[0].split(",")

            epoch = float(values[0])
            accuracy = float(values[1])
            loss = float(values[2])
            learning_rate = float(values[3])
            validation_accuracy = float(values[4])
            validation_loss = float(values[5])

            stats.append((epoch, accuracy, loss, learning_rate, validation_accuracy, validation_loss))

    return stats


def get_stats():

    stats = []

    for filename in input_path:
        stats_for_file = get_stats_from_file(filename)
        stats.append(stats_for_file)

    return stats


def main():
    gStyle.SetOptStat(0)

    stats = get_stats()
    draw_graphs(stats)

    print("Saved plots:", saved_plots)


if __name__ == "__main__":
    main()




