
//string inputPathSource = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_cmssw/qcd/scoutingAtHlt/QCD_flat_ntuples_part0.root";
//string inputPathSource = "/eos/cms/store/group/phys_exotica/svjets/backgrounds_cmssw/qcd/scoutingAtHlt/QCD_flat_ntuples_merged.root";
string inputPathSource = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_delphes/qcd/delphes/qcd_highpT_13TeV_300.root";

//string inputPathDestination = "";
//string inputPathDestination = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_delphes/qcd/delphes/qcd_highpT_13TeV_300.root";
string inputPathDestination = "/Users/Jeremi/Documents/Physics/ETH/data/s_channel_delphes/delphes/m3000_mD20_r30_alphapeak-HepMC_part-1.root";
//string inputPathDestination = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_delphes/qcd/delphes/qcd_highpT_13TeV_300.root";

string outputPath = "results/weights_qcd_realisticQCD_to_realisticSVJ_small";
//string outputPath = "results/weights_qcd_flatPtHat_to_realisticPt_small";
//string outputPath = "results/weights_qcd_flatPtHat_to_realisticPt";
//string outputPath = "results/weights_qcd_realistic_to_flatJetPt";

const int maxEvents = 10000;
const int nBins = 100;
const double maxPt = 3000;



void setupOutputPath(){
  outputPath += "_events"+to_string(maxEvents);
  outputPath += "_nBins"+to_string(nBins);
  outputPath += "_maxPt"+to_string(maxPt);
  outputPath += ".root";
}

double getJetPtWeight(double pt, TH1D *sourceDist, TH1D *destinationDist=nullptr)
{
  double sumNinput = sourceDist->GetEntries();
  double nBinsInput = sourceDist->GetNbinsX();
  double nOfPtInput = sourceDist->GetBinContent(sourceDist->GetXaxis()->FindFixBin(pt));
  
  double weight = sumNinput/(nBinsInput * nOfPtInput);
  
  if(destinationDist){
    double sumNoutput = destinationDist->GetEntries();
    double nBinsOutput = destinationDist->GetNbinsX();
    double nOfPtoutput = destinationDist->GetBinContent(destinationDist->GetXaxis()->FindFixBin(pt));
    
    weight /= sumNoutput/(nBinsOutput * nOfPtoutput);
  }
  
  return weight;
}

TTree* getTree(string path)
{
  auto inFile = TFile::Open(path.c_str());

  auto tree = (TTree*)inFile->Get("Delphes");
  bool isDelphes = true;

  if(!tree){
    tree = (TTree*)inFile->Get("Events");
    isDelphes = false;
  }

  if(!tree){
    cout<<"Couldn't find tree Delphes not Events..."<<endl;
    exit(0);
  }
  
  return tree;
}

TH1D* getPtHist(TTree *tree)
{
  TLeaf *jetPt = tree->FindLeaf("Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.pt_");
  if(!jetPt) jetPt = tree->FindLeaf("FatJet.PT");
  
  auto histJetPt = new TH1D("jet pt", "jet pt", nBins, 0, maxPt);
  
  for(int iEvent=0; iEvent<tree->GetEntries(); iEvent++){
    if(iEvent == maxEvents) break;
    tree->GetEntry(iEvent);
    if(iEvent%100==0) cout<<"Event: "<<iEvent<<endl;
    
    int nJets = jetPt->GetLen();
    for(int iJet=0; iJet<nJets; iJet++) histJetPt->Fill(jetPt->GetValue(iJet));
  }
  
  return histJetPt;
}

void produceWeightsHist()
{
  setupOutputPath();
  
  TTree *treeSource = getTree(inputPathSource);
  TH1D *histJetPtSource = getPtHist(treeSource);
  
  TH1D *histJetPtDestination = nullptr;
  
  if(inputPathDestination != ""){
    TTree *treeDestination = getTree(inputPathDestination);
    histJetPtDestination = getPtHist(treeDestination);
  }
  
  auto histJetPtWeights = new TH1D("histJetPtWeights", "histJetPtWeights", nBins, 0, maxPt);
  for(int i=0; i<histJetPtSource->GetNbinsX(); i++){
    double weight = getJetPtWeight(histJetPtSource->GetXaxis()->GetBinCenter(i), histJetPtSource, histJetPtDestination);
    histJetPtWeights->SetBinContent(i, isnormal(weight) ? weight : histJetPtWeights->GetBinContent(i-1));
  }
  
  auto outFile = new TFile(outputPath.c_str(), "recreate");
  outFile->cd();
  histJetPtWeights->Write();
  outFile->Close();
  
  cout<<"Weights stored in file: "<<outputPath<<endl;
  
}
