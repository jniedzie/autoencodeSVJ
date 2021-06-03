
string weightsPath = "results/weights_qcd_flatPtHat_to_flatJetPt.root";


//string inputPath = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_cmssw/qcd/scoutingAtHlt/QCD_flat_ntuples_part0.root";
string inputPath = "/eos/cms/store/group/phys_exotica/svjets/backgrounds_cmssw/qcd/scoutingAtHlt/QCD_flat_ntuples_merged.root";

int maxEvents = 999999999;

vector<tuple<string, int, double, double>> jetHistParams = {
  {"eta"              , 100, -3.5 , 3.5   },
  {"phi"              , 100, -3.5 , 3.5   },
  {"pt"               , 100, 0    , 3000  },
  {"pt_1"             , 100, 0    , 3000  },
  {"pt_2"             , 100, 0    , 3000  },
  {"mass"             , 100, 0    , 800   },
};

map<string, TH1D*> jetHists;

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

void prepareHists()
{
  for(auto &[name, nBins, min, max] : jetHistParams){
    jetHists[name] = new TH1D(name.c_str(), name.c_str(), nBins, min, max);
    string nameWeighted = name + "_ptWeighted";
    jetHists[nameWeighted] = new TH1D(nameWeighted.c_str(), nameWeighted.c_str(), nBins, min, max);
  }
}

void compareWeightedDists()
{
  auto weightsFile = TFile::Open(weightsPath.c_str());
  auto weightsHist = (TH1D*)weightsFile->Get("histJetPtWeights");
  
  prepareHists();
  
  auto [inputTree, isInputDelphes] = getTree(inputPath);
  
  map<string, TLeaf*> values;
  
  TLeaf *jetPt = nullptr;
  
  if(isInputDelphes){
    values["pt"] = inputTree->FindLeaf("FatJet.PT");
  }
  else{
    values["pt"] = inputTree->FindLeaf("Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.pt_");
    values["eta"] = inputTree->FindLeaf("Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.eta_");
    values["phi"] = inputTree->FindLeaf("Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.phi_");
    values["mass"] = inputTree->FindLeaf("Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.m_");
  }
  
  auto histJetPt = new TH1D("jet pt", "jet pt", 100, 0, 3000);
  auto histJetPt_weighted = new TH1D("jet pt weighted", "jet pt weighted", 100, 0, 3000);
  
  
  for(int iEvent=0; iEvent<inputTree->GetEntries(); iEvent++){
    if(iEvent == maxEvents) break;
    inputTree->GetEntry(iEvent);
    if(iEvent%100==0) cout<<"Event: "<<iEvent<<endl;
    
    int nJets = values["pt"]->GetLen();
    for(int iJet=0; iJet<nJets; iJet++){
      double pt = values["pt"]->GetValue(iJet);
      double weight = weightsHist->GetBinContent(weightsHist->GetXaxis()->FindFixBin(pt));
      
      jetHists["pt"]->Fill(pt);
      jetHists["eta"]->Fill(values["eta"]->GetValue(iJet));
      jetHists["phi"]->Fill(values["phi"]->GetValue(iJet));
      jetHists["mass"]->Fill(values["mass"]->GetValue(iJet));
      
      jetHists["pt_ptWeighted"]->Fill(pt, weight);
      jetHists["eta_ptWeighted"]->Fill(values["eta"]->GetValue(iJet), weight);
      jetHists["phi_ptWeighted"]->Fill(values["phi"]->GetValue(iJet), weight);
      jetHists["mass_ptWeighted"]->Fill(values["mass"]->GetValue(iJet), weight);
      
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
  
  TCanvas *canvas = new TCanvas("canvas", "canvas", 1280, 800);
  canvas->Divide(2, 3);
  
  int iPad=1;
  for(auto &[name, nBins, min, max] : jetHistParams){
    canvas->cd(iPad++);
    
    jetHists[name]->DrawNormalized();
    jetHists[name+"_ptWeighted"]->Sumw2(false);
    jetHists[name+"_ptWeighted"]->SetLineColor(kRed);
    jetHists[name+"_ptWeighted"]->DrawNormalized("same");
  
  }
  
  canvas->SaveAs("results/afterWeighting.pdf");
  
}
