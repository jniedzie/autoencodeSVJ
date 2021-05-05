
vector<string> inputPaths = {
//  "h5histsQCD.root",
//  "h5histsSVJ_r15.root",
//  "h5histsSVJ_r30.root",
//  "h5histsSVJ_r45.root",
//  "h5histsSVJ_r60.root",
//  "h5histsSVJ_r75.root",
  
  "h5histsSVJ_m3500_r30_delphes.root",
  "h5histsSVJ_m3500_r30_cmssw.root",
};

map<string, string> nameForPath = {
//  {"h5histsQCD.root"    , "QCD"       },
//  {"h5histsSVJ_r15.root", "SVJ r=0.15"},
//  {"h5histsSVJ_r30.root", "SVJ r=0.30"},
//  {"h5histsSVJ_r45.root", "SVJ r=0.45"},
//  {"h5histsSVJ_r60.root", "SVJ r=0.60"},
//  {"h5histsSVJ_r75.root", "SVJ r=0.75"},
  
  {"h5histsSVJ_m3500_r30_delphes.root"  , "SVJ Delphes"},
  {"h5histsSVJ_m3500_r30_cmssw.root"    , "SVJ CMSSW"},
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
  "pt", "mass", "PTD", "axis2", "EFP_1"
};

vector<int> colors = {kBlack, kRed, kOrange+1, kViolet, kBlue, kGreen+2 };

template <typename T>
string to_string_with_precision(const T a_value, const int n = 6)
{
  ostringstream out;
  out.precision(n);
  out << fixed << a_value;
  return out.str();
}

TGraph* getROCgraph(TH1D *bkgHist, TH1D *sigHist)
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
  
  return new TGraph(sigPoints.size(), &sigPoints[0], &bkgPoints[0]);
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
  
  
void compareHists()
{
  gStyle->SetOptStat(0);
  
  vector<TH2D*> histDetaDphi;
  map<string, vector<TH1D*>> hists1D;
  
  for(auto path : inputPaths){
    auto inFile = TFile::Open(path.c_str());
    
    histDetaDphi.push_back((TH2D*)inFile->Get("constituentsDetaDphi"));
    
    for(string histName : histNames1D){
      hists1D[histName].push_back((TH1D*)inFile->Get(histName.c_str()));
    }
  }
  
  
  for(auto &[name, hists] : hists1D){
    for(int i=0; i<hists.size(); i++) hists[i]->SetLineColor(colors[i]);
  }
  
  TCanvas *canvas = new TCanvas("Constituents", "Constituents", 2000, 2000);
  canvas->Divide(4, 4);
  
  TCanvas *canvas2D = new TCanvas("Constituents 2D", "Constituents 2D", 2000, 2000);
  canvas2D->Divide(4, 4);
  
  TCanvas *canvasROC = new TCanvas("ROC curves", "ROC curves", 2000, 2000);
  canvasROC->Divide(4, 4);
  
  int iPad = 1;
  
  for(int i=0; i<histDetaDphi.size(); i++){
  
    canvas2D->cd(iPad);
    
    auto hist = new TH2D(*histDetaDphi[0]);
    
    hist->Add(histDetaDphi[i], -1);
    hist->SetTitle(nameForPath[inputPaths[i]].c_str());
    hist->DrawNormalized("colz");
    
    iPad++;
  }
    
  iPad = 1;
  
  for(auto &[name, hists] : hists1D){
    auto legend = new TLegend(0.5, 0.5, 0.9, 0.9);
    
    for(int i=0; i<hists.size(); i++){
      canvas->cd(iPad);
      hists[i]->DrawNormalized(i==0 ? "" : "same");
      
      string aucString = "";
      
      if(i!=0){
        auto rocGraph = getROCgraph(hists[0], hists[i]);
        double auc = getAUC(rocGraph);
        
        if(auc < 0.5){
          rocGraph = getROCgraph(hists[i], hists[0]);
          auc = getAUC(rocGraph);
        }
        
        aucString = " (AUC: " + to_string_with_precision(auc, 3) + ")";

        rocGraph->SetMarkerStyle(20);
        rocGraph->SetMarkerSize(1.0);
        rocGraph->SetMarkerColor(colors[i]);
        rocGraph->SetTitle(name.c_str());

        canvasROC->cd(iPad);
        rocGraph->Draw(i==1 ? "AP" : "Psame");

        if(i==1){
          TF1 *fun = new TF1("fun", "x", 0, 1);
          fun->SetLineColor(kBlack);
          fun->Draw("same");
        }

      }
      
      legend->AddEntry(hists[i], (nameForPath[inputPaths[i]] + aucString).c_str(), "l");
    }
    
    canvas->cd(iPad);
    
    legend->Draw();
    
    iPad++;
  }
    

//  vector<float> signalPoints;
//  vector<float> backgroundPoints;
//
//  for(int i=0; i<hists1D["constituentsSumDr"][0]->GetNbinsX(); i++){
//
//      backgroundPoints.push_back(hists1D["constituentsSumDr"][0]->Integral(i, 100));
//
//      signalPoints.push_back(hists1D["constituentsSumDr"][1]->Integral(i, 100));
//  }
//
//  TMVA::ROCCurve *rocPlot = new TMVA::ROCCurve(signalPoints, backgroundPoints);
//  TGraph *rocGraph = rocPlot->GetROCCurve(50);

  

}
