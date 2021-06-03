
string inputPathSource = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_cmssw/qcd/scoutingAtHlt/QCD_flat_ntuples_part0.root";
//string inputPathDestination = "";
string inputPathDestination = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_delphes/qcd/delphes/qcd_highpT_13TeV_300.root";

//string outputPath = "results/weights_qcd_flatPtHat_to_flatJetPt.root";
string outputPath = "results/weights_qcd_flatPtHat_to_realistic.root";

int maxEvents = 500;

double getJetPtWeight(double pt, TH1D *inputDist, TH1D *outputDist=nullptr)
{
  double sumNinput = inputDist->GetEntries();
  double nBinsInput = inputDist->GetNbinsX();
  double nOfPtInput = inputDist->GetBinContent(inputDist->GetXaxis()->FindFixBin(pt));
  
  double weight = sumNinput/(nBinsInput * nOfPtInput);
  
  if(outputDist){
    double sumNoutput = outputDist->GetEntries();
    double nBinsOutput = outputDist->GetNbinsX();
    double nOfPtoutput = outputDist->GetBinContent(outputDist->GetXaxis()->FindFixBin(pt));
    
    weight /= sumNoutput/(nBinsOutput * nOfPtoutput);
  }
  
  return weight;
}

TH1D* getPtDist(TTree *tree, bool isDelphes)
{
  TLeaf *jetPt = nullptr;
  
  if(isDelphes) jetPt = tree->FindLeaf("FatJet.PT");
  else          jetPt = tree->FindLeaf("Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.pt_");
  
  auto histJetPt = new TH1D("jet pt", "jet pt", 100, 0, 3000);
  
  for(int iEvent=0; iEvent<tree->GetEntries(); iEvent++){
    if(iEvent == maxEvents) break;
    tree->GetEntry(iEvent);
    if(iEvent%100==0) cout<<"Event: "<<iEvent<<endl;
    
    int nJets = jetPt->GetLen();
    for(int iJet=0; iJet<nJets; iJet++) histJetPt->Fill(jetPt->GetValue(iJet));
  }
  
  return histJetPt;
}

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

void produceWeightsHist()
{
  auto [treeSource, isSourceDelphes] = getTree(inputPathSource);
  TH1D *histJetPt = getPtDist(treeSource, isSourceDelphes);
  
  TH1D *histJetPtDestination = nullptr;
  
  if(inputPathDestination != ""){
    auto [treeDest, isDestDelphes] = getTree(inputPathDestination);
    histJetPtDestination = getPtDist(treeDest, isDestDelphes);
  }
  
  auto histJetPtWeights = new TH1D("histJetPtWeights", "histJetPtWeights", 100, 0, 3000);
  
  for(int i=0; i<histJetPt->GetNbinsX(); i++){
    double weight = getJetPtWeight(histJetPt->GetXaxis()->GetBinCenter(i), histJetPt, histJetPtDestination);
    histJetPtWeights->SetBinContent(i, isnormal(weight) ? weight : 1);
  }
  
  auto outFile = new TFile(outputPath.c_str(), "recreate");
  outFile->cd();
  histJetPtWeights->Write();
  outFile->Close();
}
