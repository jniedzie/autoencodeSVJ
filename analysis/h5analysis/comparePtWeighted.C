
vector<string> variableNames = { "PTD", "axis2", "mass", "pt_1", "pt_2", "pt" };

void comparePtWeighted()
{
  gStyle->SetOptStat(0);
  
  auto inFile = TFile::Open("results/h5histsQCD_delphes_new.root");
  
  auto canvas = new TCanvas("canvas", "canvas", 1280, 800);
  canvas->Divide(2, 3);
  
  int iPad=1;
  for(auto name : variableNames){
    
    canvas->cd(iPad++);
    
    auto hist = (TH1D*)inFile->Get(name.c_str());
    auto histWeighted = (TH1D*)inFile->Get((name+"_ptWeighted").c_str());
    
    hist->SetLineColor(kBlack);
    histWeighted->SetLineColor(kRed);
    
    histWeighted->Sumw2(false);
    
    hist->DrawNormalized();
    histWeighted->DrawNormalized("same");
    
  }
    
}
