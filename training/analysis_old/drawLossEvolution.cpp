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

// archs
//string aucsPath =  "../trainingResults_archs/aucs/";
//string resultsPath =  "../trainingResults_archs/trainingRuns/";
//string filePattern = "hlf_efp_3_bottle_9_arch_60__60_loss_mean_absolute_error_batch_size_256_v";
//string filePattern = "hlf_efp_3_bottle_9_arch_19__19_loss_mean_absolute_error_batch_size_256_v";
//string filePattern = "hlf_efp_3_bottle_9_arch_19__19__19_loss_mean_absolute_error_batch_size_256_v";
//string filePattern = "hlf_efp_3_bottle_9_arch_19__19__19__19_loss_mean_absolute_error_batch_size_256_v";
//string filePattern = "hlf_efp_3_bottle_9_arch_18__15__12_loss_mean_absolute_error_batch_size_256_v";
//string filePattern = "hlf_efp_3_bottle_9_arch_19__19__9_loss_mean_absolute_error_batch_size_256_v";
//string filePattern = "hlf_efp_3_bottle_9_arch_42__42_loss_mean_absolute_error_batch_size_256_noChargedFraction_v";

// batch sizes
//string aucsPath =  "../trainingResults_batchSizes/aucs/";
//string resultsPath =  "../trainingResults_batchSizes/trainingRuns/";
//string filePattern = "hlf_efp_3_bottle_9_arch_42__42_loss_mean_absolute_error_batch_size_512_noChargedFraction_v";
//string filePattern = "hlf_efp_3_bottle_9_arch_42__42_loss_mean_absolute_error_batch_size_16_noChargedFraction_v";

// optimizers
string aucsPath =  "../trainingResults_optimizers/aucs/";
string resultsPath =  "../trainingResults_optimizers/trainingRuns/";
//string filePattern = "hlf_efp_3_bottle_9_arch_42__42_loss_mean_absolute_error_optimizer_256_batch_size_Adadelta_noChargedFraction_v";
//string filePattern = "hlf_efp_3_bottle_9_arch_42__42_loss_mean_absolute_error_optimizer_256_batch_size_Adagrad_noChargedFraction_v";
string filePattern = "hlf_efp_3_bottle_9_arch_42__42_loss_mean_absolute_error_optimizer_256_batch_size_Adamax_noChargedFraction_v";
//string filePattern = "hlf_efp_3_bottle_9_arch_42__42_loss_mean_absolute_error_optimizer_256_batch_size_Ftrl_noChargedFraction_v";
//string filePattern = "hlf_efp_3_bottle_9_arch_42__42_loss_mean_absolute_error_optimizer_256_batch_size_Nadam_noChargedFraction_v";
//string filePattern = "hlf_efp_3_bottle_9_arch_42__42_loss_mean_absolute_error_optimizer_256_batch_size_RMSprop_noChargedFraction_v";
//string filePattern = "hlf_efp_3_bottle_9_arch_42__42_loss_mean_absolute_error_optimizer_256_batch_size_SGD_noChargedFraction_v";




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

