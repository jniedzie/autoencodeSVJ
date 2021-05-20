//#include "Helpers.hpp"

const bool calculateAucs = false;

vector<string> inputPaths = {
//  "results/h5histsQCD_cmssw_mixed.root",
  "results/h5histsQCD_delphes_fixed.root",
  
//  "results/h5histsQCD_delphes.root",
//  "results/h5histsQCD_cmssw_1000to1400.root",
//  "results/h5histsQCD_cmssw_1400to1800.root",
//  "results/h5histsQCD_cmssw_1800to2400.root",
//  "results/h5histsQCD_cmssw_2400to3200.root",
//  "results/h5histsQCD_cmssw_3200toInf.root",
  
  
  
//  "h5histsSVJ_r15.root",
//  "h5histsSVJ_r30.root",
//  "h5histsSVJ_r45.root",
//  "h5histsSVJ_r60.root",
//  "h5histsSVJ_r75.root",
  
//  "results/h5histsSVJ_m3500_r30_delphes.root",
  "results/h5histsSVJ_m3500_r30_mDark40_alphaPeak_delphes.root",
  "results/h5histsSVJ_m3500_r30_mDark20_alphaPeak_delphes.root",
  "results/h5histsSVJ_m3500_r30_delphes_test.root",
  "results/h5histsSVJ_m3500_r30_cmssw.root",
  "results/h5histsSVJ_m3500_r30_mDark40_alphaPeak_cmssw_fixed.root",
};

map<string, tuple<string, int, int, int>> histParams = {
  // path                                           title                       color       style   width
//  {"results/h5histsQCD_cmssw_mixed.root"                                  , {"QCD CMSSW mixed"              , kBlue       , 1     , 1   }},
  {"results/h5histsQCD_delphes_fixed.root"                                , {"QCD Delphes"                  , kBlack      , 1     , 3   }},
  
//  {"results/h5histsSVJ_m3500_r30_delphes.root"      , {"SVJ Delphes"              , kViolet     , 2     , 3   }},
  {"results/h5histsSVJ_m3500_r30_mDark40_alphaPeak_delphes.root"          , {"SVJ Delphes mD=40, alphaPeak" , kCyan+1     , 2     , 3   }},
  {"results/h5histsSVJ_m3500_r30_mDark20_alphaPeak_delphes.root"          , {"SVJ Delphes mD=20, alphaPeak" , kGreen+1    , 2     , 3   }},
  {"results/h5histsSVJ_m3500_r30_delphes_test.root"                       , {"SVJ Delphes mD=20, alpha=0.1" , kViolet     , 2     , 3   }},
  {"results/h5histsSVJ_m3500_r30_cmssw.root"                              , {"SVJ CMSSW"                    , kRed        , 2     , 1   }},
  {"results/h5histsSVJ_m3500_r30_mDark40_alphaPeak_cmssw_fixed.root"      , {"SVJ CMSSW (fixed)"            , kBlue       , 2     , 1   }},
  
//  {"results/h5histsQCD_cmssw_1000to1400.root"   , {"QCD CMSSW 1000-1400 GeV" , kBlue      , 1 , 1 }  },
//  {"results/h5histsQCD_cmssw_1400to1800.root"   , {"QCD CMSSW 1400-1800 GeV" , kCyan      , 1 , 1 }  },
//  {"results/h5histsQCD_cmssw_1800to2400.root"   , {"QCD CMSSW 1800-2400 GeV" , kGreen     , 1 , 1 }  },
//  {"results/h5histsQCD_cmssw_2400to3200.root"   , {"QCD CMSSW 2400-3200 GeV" , kGreen+2   , 1 , 1 }  },
//  {"results/h5histsQCD_cmssw_3200toInf.root"    , {"QCD CMSSW 3200-Inf GeV"  , kOrange+1  , 1 , 1 }  },
  
//  {"h5histsSVJ_r15.root", "SVJ r=0.15", kBlack}}}},
//  {"h5histsSVJ_r30.root", "SVJ r=0.30", kBlack}}}},
//  {"h5histsSVJ_r45.root", "SVJ r=0.45", kBlack}}}},
//  {"h5histsSVJ_r60.root", "SVJ r=0.60", kBlack}}}},
//  {"h5histsSVJ_r75.root", "SVJ r=0.75", kBlack}}}},
  
  
};

vector<string> histNames1D = {
  "constituentsDr",
  "constituentsDrCorrected",
  "constituentsDrFromPrevious",
  "constituentsPt",
  "constituentsDptFromPrevious",
//  "constituentsSumDr",
//  "constituentsSumDrPrevious",
  "constituentsAvgDr",
  "constituentsAvgDrPrevious",
};

vector<string> histNamesJets = {
  "pt_1", "pt_2", "mass", "PTD", "axis2", "EFP_1"
};

vector<int> colors = {kBlack, kRed, kOrange+1, kViolet, kBlue, kGreen+2, kMagenta+1, kCyan+1 };

vector<TCanvas*> canvases;

template <typename T>
string to_string_with_precision(const T a_value, const int n = 6)
{
  ostringstream out;
  out.precision(n);
  out << fixed << a_value;
  return out.str();
}

double getAUC(TGraph *g, Int_t nb=100)
{
   Double_t  s = 0;
   Double_t *x = g->GetX();
   Double_t *y = g->GetY();
   Int_t     n = g->GetN();
   Double_t xmin = x[0], xmax = x[0];
   Double_t ymin = y[0], ymax = y[0];
   Int_t i;
                                                                                
   for (i=1; i<n; i++) {
      if (xmin>x[i]) xmin = x[i];
      if (xmax<x[i]) xmax = x[i];
      if (ymin>y[i]) ymin = y[i];
      if (ymax<y[i]) ymax = y[i];
   }
                                                                                
   Double_t dx  = (xmax-xmin)/nb;
   Double_t dy  = (ymax-ymin)/nb;
   Double_t dx2 = dx/2;
   Double_t dy2 = dy/2;
   Double_t ds  = dx*dy;
                                                                                
   Double_t xi, yi, xc, yc;
   for (xi=xmin; xi<xmax; xi=xi+dx) {
      for (yi=ymin; yi<ymax; yi=yi+dy) {
        xc = xi+dx2;
        yc = yi+dy2;
        if (TMath::IsInside(xc, yc, n, x, y)) {
           s = s + ds;
        }
      }
   }
   return s;
}

tuple<TGraph*, double> getROCgraph(TH1D *bkgHist, TH1D *sigHist)
{
  int nbins = sigHist->GetNbinsX();

  float sig_integral = sigHist->Integral(1, nbins);
  float bkg_integral = bkgHist->Integral(1, nbins);
  
  vector<float> sigPoints(nbins);
  vector<float> bkgPoints(nbins);

  sigPoints.push_back(0);
  bkgPoints.push_back(0);
  
  for ( int i = 0; i < nbins; ++i ) {
    float sig_slice_integral = sigHist->Integral(nbins-i, nbins);
    float bkg_slice_integral = bkgHist->Integral(nbins-i, nbins);
    sigPoints.push_back(sig_slice_integral/sig_integral);
    bkgPoints.push_back(bkg_slice_integral/bkg_integral);
  }
  
  sigPoints.push_back(1);
  bkgPoints.push_back(0);
  
  auto rocGraph = new TGraph(sigPoints.size(), &sigPoints[0], &bkgPoints[0]);
  double auc = getAUC(rocGraph);
  
  return make_tuple(rocGraph, auc);
}

string getAucString(TH1D *hist1, TH1D *hist2)
{
  tuple<TGraph*, double> rocs = getROCgraph(hist1, hist2);
  if(get<1>(rocs) < 0.5) rocs = getROCgraph(hist2, hist1);
  auto [rocGraph, auc] = rocs;
  return " (AUC: " + to_string_with_precision(auc, 2) + ")";
}

void drawHists(const map<string, vector<TH1D*>> &histsCollection, TCanvas *canvas)
{
  int iPad = 1;
  
  for(auto &[name, hists] : histsCollection){
    auto legend = new TLegend(0.5, 0.5, 0.9, 0.9);
    
    for(int i=0; i<hists.size(); i++){
      canvas->cd(iPad);
      hists[i]->Sumw2(false);
      hists[i]->DrawNormalized(i==0 ? "" : "same");
      string aucString = "";
      
      if(calculateAucs) string aucString = getAucString(hists[0], hists[i]);

      auto [name, color, lineStyle, lineWidth] = histParams[inputPaths[i]];
      
      legend->AddEntry(hists[i], (name + aucString).c_str(), "l");
    }
    
    canvas->cd(iPad);
    legend->Draw();
    iPad++;
  }
}

void compareHists()
{
  gStyle->SetOptStat(0);
  
  vector<TH2D*> histDetaDphi;
  map<string, vector<TH1D*>> hists1D;
  map<string, vector<TH1D*>> histsJets;
  
  cout<<"Loading histograms"<<endl;
  
  for(auto path : inputPaths){
    auto inFile = TFile::Open(path.c_str());
    if(!inFile) cout<<"Couldn't open file: "<<path<<endl;
    
    
    histDetaDphi.push_back((TH2D*)inFile->Get("constituentsDetaDphi"));
    
    auto [name, color, lineStyle, lineWidth] = histParams[path];
    
    for(string histName : histNames1D){
      hists1D[histName].push_back((TH1D*)inFile->Get(histName.c_str()));
      hists1D[histName].back()->SetLineColor(color);
      hists1D[histName].back()->SetLineStyle(lineStyle);
      hists1D[histName].back()->SetLineWidth(lineWidth);
    }
    for(string histName : histNamesJets){
      histsJets[histName].push_back((TH1D*)inFile->Get(histName.c_str()));
      histsJets[histName].back()->SetLineColor(color);
      histsJets[histName].back()->SetLineStyle(lineStyle);
      histsJets[histName].back()->SetLineWidth(lineWidth);
    }
  }
  
  
  cout<<"Plotting histograms"<<endl;
  
  TCanvas *canvas = new TCanvas("Constituents", "Constituents", 2000, 2000);
  canvas->Divide(3, 3);
  
  TCanvas *canvas2D = new TCanvas("Constituents 2D", "Constituents 2D", 2000, 2000);
  canvas2D->Divide(3, 3);
  
  TCanvas *canvasJets = new TCanvas("Jets", "Jets", 2000, 2000);
  canvasJets->Divide(2, 3);
  
  
  // plot 2d constutuents plots
  int iPad = 1;
  
  for(int i=0; i<histDetaDphi.size(); i++){
  
    canvas2D->cd(iPad);
    if(!histDetaDphi[0]) continue;
    
    auto hist = new TH2D(*histDetaDphi[0]);
    
    hist->Add(histDetaDphi[i], -1);
    
    auto [name, color, lineStyle, lineWidth] = histParams[inputPaths[i]];
    
    hist->SetTitle(name.c_str());
    hist->DrawNormalized("colz");
    
    iPad++;
  }
    
  drawHists(hists1D, canvas);
  drawHists(histsJets, canvasJets);
  
}
