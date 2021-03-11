//
//  drawLossEvolution.cpp
//  xTrainingAnalysis
//
//  Created by Jeremi Niedziela on 02/11/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#include "Helpers.hpp"

#include "Result.hpp"
#include "ResultsProcessor.hpp"

// Bottleneck:
//string aucsPath =  "../trainingResults_noLeptonVeto_fatJets_dr0p8/aucs/";
//string resultsPath =  "../trainingResults_noLeptonVeto_fatJets_dr0p8/trainingRuns/";
//string filePattern = "hlf_eflow_3_bottle_4_v";
//string filePattern = "hlf_eflow_3_bottle_5_v";
//string filePattern = "hlf_eflow_3_bottle_6_v";
//string filePattern = "hlf_eflow_3_bottle_7_v";
//string filePattern = "hlf_eflow_3_bottle_8_v";
//string filePattern = "hlf_eflow_3_bottle_9_v";
//string filePattern = "hlf_eflow_3_bottle_10_v";

// Batch size:
//string aucsPath =  "../trainingResults_noLeptonVeto_fatJets_dr0p8_batchSizes/aucs/";
//string resultsPath =  "../trainingResults_noLeptonVeto_fatJets_dr0p8_batchSizes/trainingRuns/";
//string filePattern = "hlf_eflow_3_bottle_7_bs_1_v";
//string filePattern = "hlf_eflow_3_bottle_7_bs_8_v";
//string filePattern = "hlf_eflow_3_bottle_7_bs_64_v";
//string filePattern = "hlf_eflow_3_bottle_7_bs_256_v";
//string filePattern = "hlf_eflow_3_bottle_7_bs_512_v";
//string filePattern = "hlf_eflow_3_bottle_7_bs_999999_v";
//string filePattern = "hlf_eflow_3_bottle_7_bs_999999_epochs_2000_v";

// Optimizers:
//string aucsPath =  "../trainingResults_noLeptonVeto_fatJets_dr0p8_optimizers/aucs/";
//string resultsPath =  "../trainingResults_noLeptonVeto_fatJets_dr0p8_optimizers/trainingRuns/";
//string filePattern = "hlf_eflow_3_bottle_6_optimizer_SGD_v";
//string filePattern = "hlf_eflow_3_bottle_6_optimizer_RMSprop_v";
//string filePattern = "hlf_eflow_3_bottle_6_optimizer_Adam_v";
//string filePattern = "hlf_eflow_3_bottle_6_optimizer_Adadelta_v";
//string filePattern = "hlf_eflow_3_bottle_6_optimizer_Adagrad_v";
//string filePattern = "hlf_eflow_3_bottle_6_optimizer_Adamax_v";
//string filePattern = "hlf_eflow_3_bottle_6_optimizer_Nadam_v";
//string filePattern = "hlf_eflow_3_bottle_6_optimizer_Ftrl_v";

// Losses:
string aucsPath =  "../trainingResults_noLeptonVeto_fatJets_dr0p8_losses/aucs/";
string resultsPath =  "../trainingResults_noLeptonVeto_fatJets_dr0p8_losses/trainingRuns/";
//string filePattern = "hlf_eflow_4_bottle_6_loss_mean_absolute_error_v";
//string filePattern = "hlf_eflow_4_bottle_6_loss_mean_absolute_percentage_error_v";
//string filePattern = "hlf_eflow_4_bottle_6_loss_mean_squared_error_v";
//string filePattern = "hlf_eflow_4_bottle_6_loss_mean_squared_logarithmic_error_v";
//string filePattern = "hlf_eflow_4_bottle_6_loss_huber_loss_v";
//string filePattern = "hlf_eflow_4_bottle_6_loss_log_cosh_v";
string filePattern = "hlf_eflow_4_bottle_6_loss_cosine_similarity_v";

void drawGraph(const ModelStats &stats, string title)
{
  vector<Epoch> epochsInModel = stats.epochs;
  
  TGraph *trainingLoss = new TGraph();
  TGraph *validationLoss = new TGraph();
  
  int iPoint=0;
  
  for(Epoch epoch : epochsInModel){
    trainingLoss->SetPoint(iPoint, iPoint, epoch.trainingLoss);
    validationLoss->SetPoint(iPoint, iPoint, epoch.validationLoss);
    iPoint++;
  }
  
  trainingLoss->SetMarkerStyle(20);
  trainingLoss->SetMarkerSize(0.3);
  trainingLoss->SetMarkerColor(kRed+1);
  
  validationLoss->SetMarkerStyle(20);
  validationLoss->SetMarkerSize(0.3);
  validationLoss->SetMarkerColor(kGreen+1);
    
  
  trainingLoss->Draw("AP");
  validationLoss->Draw("Psame");
  
  trainingLoss->SetTitle(title.c_str());
  trainingLoss->GetXaxis()->SetTitle("Epoch");
  trainingLoss->GetYaxis()->SetTitle("Loss");
  
  trainingLoss->GetXaxis()->SetLimits(0, 400);
//  trainingLoss->GetYaxis()->SetRangeUser(0, 0.15);
  
  TLegend *legend = new TLegend(0.5, 0.7, 0.9, 0.9);
  legend->AddEntry(trainingLoss, "training loss", "p");
  legend->AddEntry(validationLoss, "validation loss", "p");
  legend->Draw();
}

int main()
{
  cout<<"Starting drawLossEvolution"<<endl;
  gStyle->SetOptStat(0);
  useCommaAsDecimalSeparator();
 
  cout<<"Creating application"<<endl;
  TApplication app("", 0, {});
  
  cout<<"Reading results from files"<<endl;
  auto resultsProcessor = make_unique<ResultsProcessor>();
  vector<ModelStats> stats = resultsProcessor->getModelStatsFromPathMarchingPatter(aucsPath, resultsPath, filePattern);
  
  resultsProcessor->sortModelsByAucAverageOverAllSignals(stats);
  
  cout<<"The best model based on average AUC over all signals: "<<stats.front().aucsFileName<<endl;
  
  cout<<"Plotting results"<<endl;
  TCanvas *canvas = new TCanvas("c1", "c1", 600, 1000);
  canvas->Divide(1, 3);
  
  canvas->cd(1);
  drawGraph(stats.front(), "The best model");
  
  canvas->cd(2);
  drawGraph(stats[stats.size()/2.], "Average model");
  
  canvas->cd(3);
  drawGraph(stats.back(), "The worst model");
  
  canvas->Update();
  
  cout<<"Running the application"<<endl;
  app.Run();
  return 0;
}

