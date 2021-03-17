#include "Helpers.hpp"

#include "Result.hpp"
#include "ResultsProcessor.hpp"

// new results
//string aucsPath =  "../trainingResults_previous_default/aucs/";
//string resultsPath =  "../trainingResults_previous_default/trainingRuns/";
//string filePattern = "hlf_eflow_3_bottle_8_default_v";

string aucsPath =  "../trainingResults_new_default/aucs/";
string resultsPath =  "../trainingResults_new_default/trainingRuns/";
string filePattern = "hlf_eflow_4_bottle_6_new_default_v";

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

// Losses:
//string aucsPath =  "../trainingResults_noLeptonVeto_fatJets_dr0p8_losses/aucs/";
//string resultsPath =  "../trainingResults_noLeptonVeto_fatJets_dr0p8_losses/trainingRuns/";
//string filePattern = "hlf_eflow_4_bottle_6_loss_mean_absolute_error_v";
//string filePattern = "hlf_eflow_4_bottle_6_loss_mean_absolute_percentage_error_v";
//string filePattern = "hlf_eflow_4_bottle_6_loss_mean_squared_error_v";
//string filePattern = "hlf_eflow_4_bottle_6_loss_mean_squared_logarithmic_error_v";
//string filePattern = "hlf_eflow_4_bottle_6_loss_huber_loss_v";
//string filePattern = "hlf_eflow_4_bottle_6_loss_log_cosh_v";
//string filePattern = "hlf_eflow_4_bottle_6_loss_cosine_similarity_v";


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

// Variables:
//string aucsPath =  "../trainingResults_noLeptonVeto_fatJets_dr0p8_efp4/aucs/";
//string resultsPath =  "../trainingResults_noLeptonVeto_fatJets_dr0p8_efp4/trainingRuns/";
//string filePattern = "hlf_eflow_4_bottle_6_v";



string plotsTitle = "";

vector<double> r_invs = {0.15, 0.30, 0.45, 0.60, 0.75};
vector<double> masses = {1500, 2000, 2500, 3000, 3500, 4000};
vector<int> colors = {kRed, kOrange, kGreen, kGreen+2, kBlue, kViolet, kBlack};

const double xMin = 0.0;
const double xMax = 1.0;

const int nBinsRinv = 130;
const int nBinsMass = 50;

TH1D* getHistogramForVariable(const vector<ModelStats> &stats, double variableValue, bool forMass)
{
  string title = forMass ? "mass" : "r_inv";
  title += " hist" + to_string(variableValue);
  TH1D *hist = new TH1D(title.c_str(), title.c_str(), forMass ? nBinsMass : nBinsRinv, xMin, xMax);
  
  for(ModelStats stat : stats){
    
    for(Result result : stat.results){
      if((forMass ? result.mass : result.r_inv) == variableValue){
        hist->Fill(result.AUC);
      }
    }
  }
  return hist;
}

void drawHistsForVariable(const vector<ModelStats> &stats, bool forMass)
{
  TLegend *leg = new TLegend(0.1, 0.6, 0.5, 0.9);
  
  cout<<"AUCs per "<<(forMass ? "mass" : "r_inv")<<": "<<endl;
  cout<<"mean\tmeanErr\twidth\twidthErr\tmax\tmaxErr"<<endl;
  
  for(int i=0; i<(forMass ? masses.size() : r_invs.size()); i++){
    
    TH1D *hist = getHistogramForVariable(stats, forMass ? masses[i] : r_invs[i], forMass);
    
    if(i==0){
      hist->SetTitle(plotsTitle.c_str());
      hist->GetXaxis()->SetTitle("AUC");
      hist->GetYaxis()->SetTitle("# trainings");
    }
    
    hist->Sumw2();
    hist->SetLineColor(colors[i]);
    string title = forMass ? "m = " : "r_{inv} = ";
    title += to_string_with_precision(forMass ? masses[i] : r_invs[i], forMass ? 0 : 2);
    if(forMass) title += " GeV";
    leg->AddEntry(hist, title.c_str(), "l");
    
    hist->Draw(i==0 ? "" : "same");
    
    cout<<(forMass ? masses[i] : r_invs[i])<<"\t";
    
    
    cout<<hist->GetMean()<<"\t"<<hist->GetMeanError()<<"\t";
    cout<<hist->GetStdDev()<<"\t"<<hist->GetStdDevError()<<"\t";
    cout<<hist->GetXaxis()->GetBinCenter(hist->FindLastBinAbove(0))<<"\t"<<hist->GetXaxis()->GetBinWidth(0)/2.<<endl;
    
  }
  
  leg->Draw();
}

int main()
{
  cout<<"Starting drawAucDistributions"<<endl;
  gStyle->SetOptStat(0);
  useCommaAsDecimalSeparator();
 
  cout<<"Creating application"<<endl;
  TApplication app("", 0, {});
  
  cout<<"Reading results from files"<<endl;
  auto resultsProcessor = make_unique<ResultsProcessor>();
  vector<ModelStats> stats = resultsProcessor->getModelStatsFromPathMarchingPatter(aucsPath, resultsPath, filePattern);
  
  resultsProcessor->sortModelsByAucAverageOverAllSignals(stats);
  cout<<"Best model:"<<stats.front().aucsFileName<<endl;
  cout<<"Best model's avg AUC:"<<stats.front().getAucAverageOverAllSignals()<<endl;
  
  double avgAucAll = 0;
  
  for(auto stat : stats){
    avgAucAll += stat.getAucAverageOverAllSignals();
  }
  
  avgAucAll /= stats.size();
  
  cout<<"Average AUC of all trainings: "<<avgAucAll<<endl;
  
  
  cout<<"Plotting results"<<endl;
  TCanvas *canvas = new TCanvas("c1", "c1", 1000, 2000);
  canvas->Divide(1, 2);
  
  canvas->cd(1);
  drawHistsForVariable(stats, false);
    
  canvas->cd(2);
  drawHistsForVariable(stats, true);
  
  canvas->Update();
  
  cout<<"Running the application"<<endl;
  app.Run();
  return 0;
}
