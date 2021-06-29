#include "Helpers.hpp"

#include "EventProcessor.hpp"


/*
 For this to work you'll need to install HDF5 together with the C++ API. On macOS, you can do that with brew:
 
 `brew install hdf5`
 
 Then, find where the HDF5 C++ API header and libs are located and build executable with:
 
 g++ h5parser.cpp -o h5parser `root-config --libs` `root-config --cflags` -I/usr/local/Cellar/hdf5/1.12.0_1/include/ -L/usr/local/Cellar/hdf5/1.12.0_1/lib/ -lhdf5_cpp -lhdf5
 
 */

const int maxEvents = 999999999;

//const string inputPath = "/Users/Jeremi/Documents/Physics/ETH/data/training_data/qcd/base_3/data_0_data.h5";

//vector<tuple<string, double, int>> inputPaths = {
//  {"/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_delphes/qcd/h5_no_lepton_veto_fat_jets_dr0p8_efp3_fatJetstrue_constituents150_maxJets2/base_3/QCD_part_0.h5", 1, 1},
//};
//
//const string outputPath = "results/h5histsQCD_delphes_new.root";

//vector<tuple<string, double, int>> inputPaths = {
//  {"/Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/rootToH5converter/test.h5", 1, 1},
//};
//
//const string outputPath = "results/h5histsQCD_delphes_test.root";

vector<tuple<string, double, int>> inputPaths = {
  {"/Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/rootToH5converter/test_svj.h5", 1, 1},
};

const string outputPath = "results/h5histsSVJ_delphes_test.root";

//vector<tuple<string, double, int>> inputPaths = {
//  {"/Users/Jeremi/Documents/Physics/ETH/data/s_channel_delphes/h5_no_lepton_veto_fat_jets_dr0p8_efp3_fatJetstrue_constituents150_maxJets2/3000GeV_0.70_mDark20_alphaPeak/base_3/SVJ_m3000_mDark20_r70_alphaPeak.h5", 1, 1},
//};
//
//const string outputPath = "results/h5hists_SVJ_m3000_mDark20_r70_alphaPeak_delphes_new.root";

//vector<tuple<string, double, int>> inputPaths = {
//  {"/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_delphes/qcd/h5_no_lepton_veto_fat_jets_dr0p8_efp3_fatJets_150constituents_2jets/base_3/QCD_part_1.h5"  , 1, 1},
//  {"/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_delphes/qcd/h5_no_lepton_veto_fat_jets_dr0p8_efp3_fatJets_150constituents_2jets/base_3/QCD_part_12.h5" , 1, 1},
//};
//
//const string outputPath = "results/h5histsQCD_delphes_fixed.root";



//vector<tuple<string, double, int>> inputPaths = {
//  {"/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_cmssw/qcd/h5_no_lepton_veto_fat_jets_dr0p8_withConstituents/base_3/QCD_pt3200toInf.h5", 1, 1},
//};
//
//const string outputPath = "results/h5histsQCD_cmssw_3200toInf.root";

//vector<tuple<string, double, int>> inputPaths = {
//  //  path                                                                                                                                        x-sec (pb)  n_generated
//  {"/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_cmssw/qcd/h5_no_lepton_veto_fat_jets_dr0p8_withConstituents/base_3/QCD_pt1000to1400.h5"  , 7.398     , 19967700  },
//  {"/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_cmssw/qcd/h5_no_lepton_veto_fat_jets_dr0p8_withConstituents/base_3/QCD_pt1400to1800.h5"  , 6.42E-01  , 5434800   },
//  {"/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_cmssw/qcd/h5_no_lepton_veto_fat_jets_dr0p8_withConstituents/base_3/QCD_pt1800to2400.h5"  , 0.08671   , 2999700   },
//  {"/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_cmssw/qcd/h5_no_lepton_veto_fat_jets_dr0p8_withConstituents/base_3/QCD_pt2400to3200.h5"  , 0.005193  , 1919400   },
//  {"/Users/Jeremi/Documents/Physics/ETH/data/backgrounds_cmssw/qcd/h5_no_lepton_veto_fat_jets_dr0p8_withConstituents/base_3/QCD_pt3200toInf.h5"   , 0.000134  , 800000    },
//};
//
//const string outputPath = "results/h5histsQCD_cmssw_mixed.root";


//const string inputPath = "/Users/Jeremi/Documents/Physics/ETH/data/s_channel_delphes/h5_no_lepton_veto_fat_jets_dr0p8_withConstituents/3500GeV_0.30/base_3/SVJ_m3500_r30.h5";
//const string outputPath = "results/h5histsSVJ_m3500_r30_delphes.root";

map<string, TH1D*>  eventHists;
map<string, TH1D*>  jetHists;
map<string, TH1D*>  constituentHists;
map<int, TH1D*>     EFPhists;
map<string, TH2D*>  hists2d;

TFile *outfile;

vector<tuple<string, int, double, double>> eventHistParams = {
  {"MET"    , 100, 0    , 5000  },
  {"METeta" , 100, -4   , 4     },
  {"METphi" , 100, -4   , 4     },
  {"MT"     , 100, 0    , 5000  },
  {"Mjj"    , 100, 0    , 5000  },
  {"nJets"  , 20 , 0    , 20    },
  {"sumJetPt", 100, 0   , 10000 },
};

vector<tuple<string, int, double, double>> jetHistParams = {
  {"eta"              , 100, -3.5 , 3.5   },
  {"phi"              , 100, -3.5 , 3.5   },
  {"pt"               , 100, 0    , 3000  },
  {"pt_1"             , 100, 0    , 3000  },
  {"pt_2"             , 100, 0    , 3000  },
  {"mass"             , 100, 0    , 800   },
  {"chargedFraction"  , 100, 0    , 1     },
  {"PTD"              , 100, 0    , 1     },
  {"axisMinor"        , 100, 0    , 0.5   },
  {"axisMajor"        , 100, 0    , 0.5   },
  {"girth"            , 100, 0    , 1.0   },
  {"lha"              , 100, 0    , 1.0   },
  {"flavor"           , 100, -50  , 50    },
  {"energy"           , 100, 0    , 5000  },
  {"efpMass"          , 100, 0    , 500   },
  {"e2"               , 100, 0    , 1.0   },
  {"e3"               , 100, 0    , 1.0   },
  {"C2"               , 100, 0    , 1.0   },
  {"D2"               , 100, 0    , 100   },
};

vector<tuple<string, int, double, double>> constituentHistParams = {
  {"constituentsDr"               , 200   , 0   , 1   },
  {"constituentsDrCorrected"      , 1000  , 0   , 0.1 },
  {"constituentsDrFromPrevious"   , 200   , 0   , 2   },
  {"constituentsPt"               , 200   , 0   , 10  },
  {"constituentsDptFromPrevious"  , 200   , -4  , 0   },
  {"constituentsSumDr"            , 100   , 0   , 100 },
  {"constituentsSumDrPrevious"    , 100   , 0   , 100 },
  {"constituentsAvgDr"            , 100   , 0.1 , 0.6 },
  {"constituentsAvgDrPrevious"    , 100   , 0.2 , 0.8 },
  {"nConstituents"                , 500   , 0   , 500 },
};

vector<tuple<string, int, double, double, int, double, double>> hists2Dparams = {
  {"efpVErification"              , 50, 0, 200, 50, 0, 200},
  {"constituentsDetaDphi"         , 1000, -1, 1, 1000, -1, 1},
};

vector<shared_ptr<Event>> getEventsFromFile(string inputPath)
{
  auto eventProcessor = make_unique<EventProcessor>();
  
  // Open H5 file, get data groups
  H5File file(inputPath.c_str(), H5F_ACC_RDONLY);
  Group rootGroup = file.openGroup("/");
  Group eventFeaturesGroup    = rootGroup.openGroup("event_features");
  Group jetEFPsGroup          = rootGroup.openGroup("jet_eflow_variables");
  Group jetFeaturesGroup      = rootGroup.openGroup("jet_features");
  Group jetConstituentsGroup  = rootGroup.openGroup("jet_constituents");
  
  // Load data into vector of events
  return eventProcessor->getValues(eventFeaturesGroup, jetEFPsGroup, jetFeaturesGroup, jetConstituentsGroup);
}

void fillEventHists(const shared_ptr<Event> event, int nJets, double eventWeight, string suffix="")
{
  eventHists["MET"+suffix]->Fill(event->MET, eventWeight);
  eventHists["METeta"+suffix]->Fill(event->METeta, eventWeight);
  eventHists["METphi"+suffix]->Fill(event->METphi, eventWeight);
  eventHists["MT"+suffix]->Fill(event->MT, eventWeight);
  eventHists["Mjj"+suffix]->Fill(event->Mjj, eventWeight);
  eventHists["nJets"+suffix]->Fill(nJets, eventWeight);
  eventHists["sumJetPt"+suffix]->Fill(event->getSumJetPt(), eventWeight);
}

void fillJetHists(const shared_ptr<Jet> jet, int iJet, double eventWeight, string suffix="")
{
  jetHists["eta"+suffix]->Fill(jet->eta, eventWeight);
  jetHists["phi"+suffix]->Fill(jet->phi, eventWeight);
  jetHists["pt"+suffix]->Fill(jet->pt, eventWeight);
  if(iJet==0) jetHists["pt_1"+suffix]->Fill(jet->pt, eventWeight);
  if(iJet==1) jetHists["pt_2"+suffix]->Fill(jet->pt, eventWeight);
  jetHists["mass"+suffix]->Fill(jet->mass, eventWeight);
  jetHists["chargedFraction"+suffix]->Fill(jet->chargedFraction, eventWeight);
  jetHists["PTD"+suffix]->Fill(jet->PTD, eventWeight);
  jetHists["axisMinor"+suffix]->Fill(jet->axisMinor, eventWeight);
  jetHists["axisMajor"+suffix]->Fill(jet->axisMajor, eventWeight);
  jetHists["girth"+suffix]->Fill(jet->girth, eventWeight);
  jetHists["lha"+suffix]->Fill(jet->lha, eventWeight);
  jetHists["flavor"+suffix]->Fill(jet->flavor, eventWeight);
  jetHists["energy"+suffix]->Fill(jet->energy, eventWeight);
  jetHists["e2"+suffix]->Fill(jet->e2, eventWeight);
  jetHists["e3"+suffix]->Fill(jet->e3, eventWeight);
  jetHists["C2"+suffix]->Fill(jet->C2, eventWeight);
  jetHists["D2"+suffix]->Fill(jet->D2, eventWeight);
  
  double recoMass = jet->pt * sqrt(0.1*jet->EFPs[1]/2);
  hists2d["efpVErification"]->Fill(jet->mass, recoMass, eventWeight);
  jetHists["efpMass"+suffix]->Fill(recoMass, eventWeight);
}

void fillEfpHists(const shared_ptr<Jet> jet, double eventWeight)
{
  for(int iEFP=0; iEFP<13; iEFP++){
    EFPhists[iEFP]->Fill(jet->EFPs[iEFP], eventWeight);
  }
}

void fillConstituentHists(const shared_ptr<Jet> jet, double eventWeight, string suffix="")
{
  shared_ptr<Constituent> previousConstituent = nullptr;
  
  TLorentzVector jetVector;
  jetVector.SetPtEtaPhiM(jet->pt, jet->eta, jet->phi, jet->mass);
  
  double sumDr = 0;
  double sumDrPrevious = 0;
  int nSumDr = 0;
  int nSumDrPrevious = 0;
  
  int nConstit = 0;
  
  for(auto constituent : jet->constituents){
    if(constituent->isEmpty()) continue;
    
    TLorentzVector constituentVector;
    constituentVector.SetPtEtaPhiE(constituent->pt, constituent->eta, constituent->phi, constituent->energy);
    
    double dEta = jet->eta - constituent->eta;
    double dPhi = jetVector.DeltaPhi(constituentVector);
    double dR = jetVector.DeltaR(constituentVector);
    
    hists2d["constituentsDetaDphi"]->Fill(dEta, dPhi, eventWeight);
    constituentHists["constituentsDr"+suffix]->Fill(dR, eventWeight);
    
    sumDr += dR;
    nSumDr++;
    
    constituentHists["constituentsDrCorrected"+suffix]->Fill(dR, eventWeight);
    constituentHists["constituentsPt"+suffix]->Fill(constituent->pt, eventWeight);
    
    if(previousConstituent){
      TLorentzVector previousConstituentVector;
      previousConstituentVector.SetPtEtaPhiE(previousConstituent->pt, previousConstituent->eta,
                                             previousConstituent->phi, previousConstituent->energy);
      
      double dPtPrevious  = constituent->pt - previousConstituent->pt;
      double dRprevious = constituentVector.DeltaR(previousConstituentVector);
      
      constituentHists["constituentsDrFromPrevious"+suffix]->Fill(dRprevious, eventWeight);
      sumDrPrevious += dRprevious;
      nSumDrPrevious++;
      
      constituentHists["constituentsDptFromPrevious"+suffix]->Fill(dPtPrevious, eventWeight);
    }
    
    previousConstituent = constituent;
    nConstit++;
  }
  
  constituentHists["constituentsSumDr"+suffix]->Fill(sumDr, eventWeight);
  constituentHists["constituentsSumDrPrevious"+suffix]->Fill(sumDrPrevious, eventWeight);
  constituentHists["constituentsAvgDr"+suffix]->Fill(sumDr/nSumDr, eventWeight);
  constituentHists["constituentsAvgDrPrevious"+suffix]->Fill(sumDrPrevious/nSumDrPrevious, eventWeight);
  constituentHists["nConstituents"+suffix]->Fill(nConstit, eventWeight);
}

void produceCorrectedHists()
{
  double lastValue = 0;
  
  for(int iBin=0; iBin<constituentHists["constituentsDrCorrected"]->GetNbinsX(); iBin++){
    double binContent = constituentHists["constituentsDrCorrected"]->GetBinContent(iBin);
    double binCenter = constituentHists["constituentsDrCorrected"]->GetXaxis()->GetBinCenter(iBin);
    
    double expectedContent = 2*TMath::Pi()*binCenter;
    constituentHists["constituentsDrCorrected"]->SetBinContent(iBin, binContent - expectedContent);
    
    if(binCenter < 0.8) lastValue = binContent - expectedContent;
  }
  
  for(int iBin=0; iBin<constituentHists["constituentsDrCorrected"]->GetNbinsX(); iBin++){
    double binContent = constituentHists["constituentsDrCorrected"]->GetBinContent(iBin);
    double binCenter = constituentHists["constituentsDrCorrected"]->GetXaxis()->GetBinCenter(iBin);
    constituentHists["constituentsDrCorrected"]->SetBinContent(iBin, binCenter < 0.8 ? binContent - lastValue : 0.0);
  }
}

void initHists()
{
  for(auto &[name, nBins, min, max] : eventHistParams){
    eventHists[name] = new TH1D(name.c_str(), name.c_str(), nBins, min, max);
    name += "_ptWeighted";
    eventHists[name] = new TH1D(name.c_str(), name.c_str(), nBins, min, max);
  }
  
  for(auto &[name, nBins, min, max] : jetHistParams){
    jetHists[name] = new TH1D(name.c_str(), name.c_str(), nBins, min, max);
    name += "_ptWeighted";
    jetHists[name] = new TH1D(name.c_str(), name.c_str(), nBins, min, max);
  }
  
  for(auto &[name, nBins, min, max] : constituentHistParams){
    constituentHists[name] = new TH1D(name.c_str(), name.c_str(), nBins, min, max);
    name += "_ptWeighted";
    constituentHists[name] = new TH1D(name.c_str(), name.c_str(), nBins, min, max);
  }
  
  for(int i=0; i<13; i++){
    string name = "EFP_"+to_string(i);
    EFPhists[i] = new TH1D(name.c_str(), name.c_str(), 100, 0, 1.0);
  }
  
  for(auto &[name, nBinsX, minX, maxX, nBinsY, minY, maxY] : hists2Dparams){
    hists2d[name] = new TH2D(name.c_str(), name.c_str(), nBinsX, minX, maxX, nBinsY, minY, maxY);
  }
}

void plotAndSaveHists()
{
  TCanvas *canvasEvents = new TCanvas("Events", "Events", 1000, 1500);
  canvasEvents->Divide(2, 3);
  
  int iPad=1;
  for(auto &[name, hist] : eventHists){
    canvasEvents->cd(iPad++);
    hist->DrawNormalized();
    outfile->cd();
    hist->Write();
  }
  
  TCanvas *canvasJets = new TCanvas("Jets", "Jets", 2880, 1800);
  canvasJets->Divide(7, 3);
  
  iPad=1;
  canvasJets->cd(iPad++);
  hists2d["efpVErification"]->Draw("colz");
  
  for(auto &[name, hist] : jetHists){
    if(name.find("_ptWeighted") == string::npos){
      canvasJets->cd(iPad++);
      hist->DrawNormalized();
      
      if(name=="mass"){
        jetHists["efpMass"]->SetLineColor(kRed);
        jetHists["efpMass"]->Draw("same");
      }
    }
    outfile->cd();
    hist->Write();
  }
  
  TCanvas *canvasEFPs = new TCanvas("EFPs", "EFPs", 2000, 1500);
  canvasEFPs->Divide(4, 3);
  
  iPad=1;
  for(auto &[iEFP, hist] : EFPhists){
    canvasEFPs->cd(iPad++);
    hist->DrawNormalized();
    outfile->cd();
    hist->Write();
  }
  
  gStyle->SetOptStat(0);
  
  TCanvas *canvasConstituents = new TCanvas("Constituents", "Constituents", 2000, 2000);
  canvasConstituents->Divide(4, 3);
  
  iPad=1;
  
  canvasConstituents->cd(iPad++);
  hists2d["constituentsDetaDphi"]->Draw("colz");
  hists2d["constituentsDetaDphi"]->GetZaxis()->SetRangeUser(0, 200);
  
  for(auto &[name, hist] : constituentHists){
    canvasConstituents->cd(iPad++);
    hist->Draw();
    outfile->cd();
    hist->Write();
  }
  
  canvasEvents->Update();
  canvasJets->Update();
  canvasEFPs->Update();
  canvasConstituents->Update();

}

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

int main (int argc, char** argv)
{
  TApplication theApp("App",&argc, argv);
  
  initHists();
  
  int iEvent=0;
  
  for(auto &[inputPath, crossSection, nGenEvents] : inputPaths){
    
    auto events = getEventsFromFile(inputPath);
    double sumGenWeights = 0;
    
    cout<<"Looping over events..."<<endl;
    
    for(auto event : events){
      
      if(iEvent >= maxEvents) break;
      iEvent++;
      
      double eventWeight = crossSection * event->genWeight/nGenEvents;
      sumGenWeights += event->genWeight;
      
      int iJet = 0;
      for(auto jet : event->jets){
        if(jet->isEmpty()) continue;
        
        fillJetHists(jet, iJet, eventWeight);
        fillEfpHists(jet, eventWeight);
        fillConstituentHists(jet, eventWeight);
        
        iJet++;
      }
      
      fillEventHists(event, iJet, eventWeight);
    }
    cout<<"path: "<<inputPath<<"\tsum gen weights: "<<sumGenWeights<<endl;
    cout<<"n events: "<<events.size()<<endl;
  }
  
  produceCorrectedHists();
  
  for(auto &[inputPath, crossSection, nGenEvents] : inputPaths){
    
    auto events = getEventsFromFile(inputPath);
    
    cout<<"Looping over events..."<<endl;
    
    for(auto event : events){
      
      if(iEvent >= maxEvents) break;
      iEvent++;
      
      double eventWeight = crossSection * event->genWeight/nGenEvents;
      
      
      int iJet = 0;
      for(auto jet : event->jets){
        if(jet->isEmpty()) continue;
        
        double jetWeight = getJetPtWeight(jet->pt, jetHists["pt"]);
        
        fillJetHists(jet, iJet, eventWeight * jetWeight, "_ptWeighted");
        fillConstituentHists(jet, eventWeight * jetWeight, "_ptWeighted");
        
        iJet++;
      }
    }
  }
  
  cout<<"Processed "<<iEvent<<" events"<<endl;
  
  cout<<"Plotting and saving histograms..."<<endl;
  outfile = new TFile(outputPath.c_str(), "recreate");
  plotAndSaveHists();
  
  
  auto jetWeightsHist = new TH1D("jetWeightsHist", "jetWeightsHist", 100, 0, 2000);
  
  for(int i=0; i<jetHists["pt"]->GetNbinsX(); i++){
    
    double weight = getJetPtWeight(jetHists["pt"]->GetXaxis()->GetBinCenter(i), jetHists["pt"]);
    jetWeightsHist->SetBinContent(i, isnormal(weight) ? weight : 1);
  }
  outfile->cd();
  jetWeightsHist->Write();
  
  outfile->Close();
  cout<<"Plotting and saving done"<<endl;
  
  
  theApp.Run();
  return 0;
}
