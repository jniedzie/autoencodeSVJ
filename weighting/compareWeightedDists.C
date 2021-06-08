#include "TMath.h"


//string weightsPath = "results/weights_qcd_flatPtHat_to_flatJetPt.root";
//string weightsPath = "results/weights_qcd_flatPtHat_to_realisticJetPt_small_events10000_nBins100_maxPt3000.000000.root";
string weightsPath = "results/weights_qcd_realisticQCD_to_realisticSVJ_small_events10000_nBins100_maxPt3000.000000.root";
//string weightsPath = "results/weights_qcd_realistic_to_flatJetPt_events10000_nBins100_maxPt3000.000000.root";


//string inputPath = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_cmssw/qcd/scoutingAtHlt/QCD_flat_ntuples_part0.root";
//string inputPath = "/eos/cms/store/group/phys_exotica/svjets/backgrounds_cmssw/qcd/scoutingAtHlt/QCD_flat_ntuples_merged.root";
string inputPath = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_delphes/qcd/delphes/qcd_highpT_13TeV_300.root";

//string referencePath = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_delphes/qcd/delphes/qcd_highpT_13TeV_300.root";
string referencePath = "/Users/Jeremi/Documents/Physics/ETH/data/s_channel_delphes/delphes/m3000_mD20_r30_alphapeak-HepMC_part-1.root";

int maxEvents = 10000;
double minFitPt = 200;
double maxFitPt = 3000;
bool useFitFunction = false;
int polyOrder = 10;

vector<tuple<string, int, double, double>> jetHistParams = {
  {"eta"              , 100, -3.5 , 3.5   },
  {"phi"              , 100, -3.5 , 3.5   },
  {"pt"               , 100, 0    , 3000  },
  {"pt_1"             , 100, 0    , 3000  },
  {"pt_2"             , 100, 0    , 3000  },
  {"mass"             , 100, 0    , 800   },
};

map<string, TH1D*> jetHists;
map<string, TH1D*> jetHistsReference;

tuple<TTree*, bool> getTree(string path)
{
  auto inFile = TFile::Open(path.c_str());
  

  auto tree = (TTree*)inFile->Get("Delphes");
  bool isDelphes = true;

  if(!tree){
    tree = (TTree*)inFile->Get("Events");
    isDelphes = false;
  }

  if(!tree){
    cout<<"Couldn't find tree Delphes nor Events..."<<endl;
    exit(0);
  }
  
  return {tree, isDelphes};
}

TF1* getFitFunction()
{
  string fun = "[0]+[1]*x";
  for(int iOrder=2; iOrder<polyOrder; iOrder++) fun += "+["+to_string(iOrder)+"]*pow(x, "+to_string(iOrder)+")";

//  fun += "+["+to_string(polyOrder)+"]/(["+to_string(polyOrder+1)+"]*sqrt(2*TMath::Pi())) * exp(-0.5  * pow( (x-["+to_string(polyOrder+2)+"])/["+to_string(polyOrder+1)+"], 2))";
//
  auto fitFun = new TF1("fit fun", fun.c_str(), minFitPt, 10000);
  fitFun->SetParameter(0, 0);
  
//  fitFun->SetParameter(polyOrder, 2); // gauss height
//  fitFun->SetParameter(polyOrder+1, 80); // sigma
//  fitFun->SetParLimits(polyOrder+1, 20, 200);
//  fitFun->SetParameter(polyOrder+2, 460); // mu
//  fitFun->SetParLimits(polyOrder+2, 200, 600);
  

  for(int iOrder=1; iOrder<polyOrder; iOrder++){
    fitFun->SetParameter(iOrder, iOrder <8 ? 1 : 0);
  }

//  auto fitFun = new TF1("fit fun", "[0]+[5]/([1]*sqrt(2*TMath::Pi())) * exp(-0.5  * pow( (x-[2])/[1], 2)) + [3] *exp ([4]*x)", minFitPt, 10000);
//  auto fitFun = new TF1("fit fun", "[0]+[5]/([1]*sqrt(2*TMath::Pi())) * exp(-0.5  * pow( (x-[2])/[1], 2)) + [3]*pow(x, [4])", minFitPt, 10000);
//  fitFun->SetParameter(0, 1);
//  fitFun->SetParameter(1, 80); // sigma
//  fitFun->SetParLimits(1, 20, 200);
//  fitFun->SetParameter(2, 460); // mu
//  fitFun->SetParLimits(2, 200, 600);
//  fitFun->SetParameter(5, 2); // gauss height
//  fitFun->SetParameter(3, 0);
//  fitFun->FixParameter(4, 3);
//
  return fitFun;
}

void prepareHists()
{
  for(auto &[name, nBins, min, max] : jetHistParams){
    jetHists[name] = new TH1D(name.c_str(), name.c_str(), nBins, min, max);
    jetHistsReference[name] = new TH1D(name.c_str(), name.c_str(), nBins, min, max);
    string nameWeighted = name + "_ptWeighted";
    jetHists[nameWeighted] = new TH1D(nameWeighted.c_str(), nameWeighted.c_str(), nBins, min, max);
  }
}

map<string, TLeaf*> getValues(TTree *tree, bool isDelphes)
{
  map<string, TLeaf*> values;
  
  if(isDelphes){
    values["pt"] = tree->FindLeaf("FatJet.PT");
    values["eta"] = tree->FindLeaf("FatJet.Eta");
    values["phi"] = tree->FindLeaf("FatJet.Phi");
    values["mass"] = tree->FindLeaf("FatJet.Mass");
  }
  else{
    values["pt"] = tree->FindLeaf("Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.pt_");
    values["eta"] = tree->FindLeaf("Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.eta_");
    values["phi"] = tree->FindLeaf("Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.phi_");
    values["mass"] = tree->FindLeaf("Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.m_");
  }
  
  return values;
}

void fillHists(TTree *tree, const map<string, TLeaf*> &values, TH1D *weightsHist, TF1 *fitFun=nullptr)
{
  for(int iEvent=0; iEvent<tree->GetEntries(); iEvent++){
    if(iEvent == maxEvents) break;
    tree->GetEntry(iEvent);
    if(iEvent%100==0) cout<<"Event: "<<iEvent<<endl;
    
    int nJets = values.at("pt")->GetLen();
    for(int iJet=0; iJet<nJets; iJet++){
      double pt = values.at("pt")->GetValue(iJet);
      double weight = 1.0;
      
      if(fitFun)  weight = fitFun->Eval(pt);
      else        weight = weightsHist->GetBinContent(weightsHist->GetXaxis()->FindFixBin(pt));
      
      jetHists["pt"]->Fill(pt);
      jetHists["eta"]->Fill(values.at("eta")->GetValue(iJet));
      jetHists["phi"]->Fill(values.at("phi")->GetValue(iJet));
      jetHists["mass"]->Fill(values.at("mass")->GetValue(iJet));
      
      jetHists["pt_ptWeighted"]->Fill(pt, weight);
      jetHists["eta_ptWeighted"]->Fill(values.at("eta")->GetValue(iJet), weight);
      jetHists["phi_ptWeighted"]->Fill(values.at("phi")->GetValue(iJet), weight);
      jetHists["mass_ptWeighted"]->Fill(values.at("mass")->GetValue(iJet), weight);
      
      if(iJet==0){
        jetHists["pt_1"]->Fill(pt);
        jetHists["pt_1_ptWeighted"]->Fill(pt, weight);
      }
      if(iJet==1){
        jetHists["pt_2"]->Fill(pt);
        jetHists["pt_2_ptWeighted"]->Fill(pt, weight);
      }
    }
  }
}

void fillReferenceHists(TTree *tree, const map<string, TLeaf*> &values)
{
  for(int iEvent=0; iEvent<tree->GetEntries(); iEvent++){
    if(iEvent == maxEvents) break;
    tree->GetEntry(iEvent);
    if(iEvent%100==0) cout<<"Event: "<<iEvent<<endl;
    
    int nJets = values.at("pt")->GetLen();
    for(int iJet=0; iJet<nJets; iJet++){
      jetHistsReference["pt"]->Fill(values.at("pt")->GetValue(iJet));
      jetHistsReference["eta"]->Fill(values.at("eta")->GetValue(iJet));
      jetHistsReference["phi"]->Fill(values.at("phi")->GetValue(iJet));
      jetHistsReference["mass"]->Fill(values.at("mass")->GetValue(iJet));
      
      if(iJet==0) jetHistsReference["pt_1"]->Fill(values.at("pt")->GetValue(iJet));
      if(iJet==1) jetHistsReference["pt_2"]->Fill(values.at("pt")->GetValue(iJet));
    }
  }
}

void compareWeightedDists()
{
  auto weightsFile = TFile::Open(weightsPath.c_str());
  auto weightsHist = (TH1D*)weightsFile->Get("histJetPtWeights");
  
  TF1 *fitFun = nullptr;
  
  if(useFitFunction){
    fitFun = getFitFunction();
    weightsHist->Fit(fitFun, "", "", minFitPt, maxFitPt);

    cout<<"Fit chi2: "<<fitFun->GetChisquare()<<endl;
    cout<<"Fit chi2/ndf: "<<fitFun->GetChisquare()/(weightsHist->GetNbinsX()-polyOrder+1)<<endl;
  }
  
  prepareHists();
  
  // Prepare input and weighted hists
  auto [inputTree, isInputDelphes] = getTree(inputPath);
  map<string, TLeaf*> values = getValues(inputTree, isInputDelphes);
  fillHists(inputTree, values, weightsHist, fitFun);
  
  // Prepare reference hists
  auto [referenceTree, isReferenceDelphes] = getTree(referencePath);
  map<string, TLeaf*> valuesReference = getValues(referenceTree, isReferenceDelphes);
  fillReferenceHists(referenceTree, valuesReference);
  
  TCanvas *canvasWeights = new TCanvas("canvasWeights", "canvasWeights", 800, 600);
  canvasWeights->Divide(1, 2);
  
  canvasWeights->cd(1);
  weightsHist->Draw();
  
  if(useFitFunction){
    canvasWeights->cd(2);
    fitFun->Draw();
  }
  
  
  auto canvas = new TCanvas("canvas", "canvas", 1280, 800);
  canvas->Divide(2, 3);
  
  auto legend = new TLegend(0.7, 0.7, 0.9, 0.9);
  
  int iPad=1;
  for(auto &[name, nBins, min, max] : jetHistParams){
    
    canvas->cd(iPad++);
    
    jetHists[name]->DrawNormalized();
    
    jetHists[name+"_ptWeighted"]->Sumw2(false);
    jetHists[name+"_ptWeighted"]->SetLineColor(kRed);
    jetHists[name+"_ptWeighted"]->DrawNormalized("same");
  
    jetHistsReference[name]->Sumw2(false);
    jetHistsReference[name]->SetLineColor(kGreen);
    jetHistsReference[name]->DrawNormalized("same");
    
    if(iPad==2){
      legend->AddEntry(jetHists[name], "input", "l");
      legend->AddEntry(jetHists[name+"_ptWeighted"], "weighted", "l");
      legend->AddEntry(jetHistsReference[name], "target", "l");
    }
  }
  
  canvas->cd(1);
  legend->Draw();
  
  canvas->SaveAs("results/afterWeighting.pdf");
  
}
