#include "Helpers.hpp"

#include "EventProcessor.hpp"


/*
 For this to work you'll need to install HDF5 together with the C++ API. On macOS, you can do that with brew:
 
 `brew install hdf5`
 
 Then, find where the HDF5 C++ API header and libs are located and build executable with:
 
 g++ h5parser.cpp -o h5parser `root-config --libs` `root-config --cflags` -I/usr/local/Cellar/hdf5/1.12.0_1/include/ -L/usr/local/Cellar/hdf5/1.12.0_1/lib/ -lhdf5_cpp -lhdf5
 
 */

//const string inputPath = "/Users/Jeremi/Documents/Physics/ETH/data/training_data/qcd/base_3/data_0_data.h5";

//const string inputPath = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds/qcd/h5_no_lepton_veto_fat_jets_dr0p8_withConstituents/base_3/QCD_part_1.h5";
//const string outputPath = "h5histsQCD.root";

const string inputPath = "/Users/Jeremi/Documents/Physics/ETH/data/s_channel_delphes/h5_no_lepton_veto_fat_jets_dr0p8_withConstituents/3000GeV_0.15/base_3/SVJ_m3000_r15.h5";
const string outputPath = "h5histsSVJ_r15.root";

//const string inputPath = "../../rootToH5converter/test.h5";

//const string inputPath = "/Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/rootToH5converter/test_data_0.h5";

vector<tuple<string, int, double, double>> eventHistParams = {
  {"MET"    , 100, 0    , 5000  },
  {"METeta" , 100, -4   , 4     },
  {"METphi" , 100, -4   , 4     },
  {"MT"     , 100, 0    , 5000  },
  {"Mjj"    , 100, 0    , 5000  },
};

vector<tuple<string, int, double, double>> jetHistParams = {
  {"eta"              , 100, -3.5   , 3.5     },
  {"phi"              , 100, -3.5   , 3.5     },
  {"pt"               , 100, 0    , 2000  },
  {"mass"             , 100, 0    , 800   },
  {"chargedFraction"  , 100, 0    , 1     },
  {"PTD"              , 100, 0    , 1     },
  {"axis2"            , 100, 0    , 0.2   },
  {"flavor"           , 100, -50  , 50    },
  {"energy"           , 100, 0    , 5000  },
};

int main (int argc, char** argv)
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
  auto events = eventProcessor->getValues(eventFeaturesGroup, jetEFPsGroup, jetFeaturesGroup, jetConstituentsGroup);
  
  
  TApplication theApp("App",&argc, argv);
  
  map<string, TH1D*>  eventHists;
  map<string, TH1D*>  jetHists;
  map<int, TH1D*>     EFPhists;
  
  for(auto &[name, nBins, min, max] : eventHistParams){
    eventHists[name] = new TH1D(name.c_str(), name.c_str(), nBins, min, max);
  }
  
  for(auto &[name, nBins, min, max] : jetHistParams){
    jetHists[name] = new TH1D(name.c_str(), name.c_str(), nBins, min, max);
  }
  
  for(int i=0; i<13; i++){
    string name = "EFP_"+to_string(i);
    EFPhists[i] = new TH1D(name.c_str(), name.c_str(), 100, 0, 1.0);
  }
  
  TH2D *jetsMasses = new TH2D("jetsMasses", "jetsMasses", 100, 0, 3500, 100, 0, 500);
  jetsMasses->GetXaxis()->SetTitle("Event Mjj (GeV)");
  jetsMasses->GetYaxis()->SetTitle("m_{j1}+m_{j2} (GeV)");
  
  TH2D *efpVErification = new TH2D("EFP verification", "EFP verification", 50, 0, 200, 50, 0, 200);
  TH1D *efpMAss = new TH1D("EFP mass", "EFP mass", 100, 0, 500);
  efpMAss->SetLineColor(kRed);
  
  TH2D *constituentsDetaDphi        = new TH2D("constituentsDetaDphi"       , "constituentsDetaDphi", 200, -0.8, 0.8, 200, -0.8, 0.8);
  TH1D *constituentsDr              = new TH1D("constituentsDr"             , "constituentsDr"              , 200, 0, 1);
  TH1D *constituentsDrCorrected     = new TH1D("constituentsDrCorrected"    , "constituentsDrCorrected"     , 1000, 0, 0.1);
  
  TH1D *constituentsDrFromPrevious  = new TH1D("constituentsDrFromPrevious" , "constituentsDrFromPrevious"  , 200, 0, 2);
  TH1D *constituentsPt              = new TH1D("constituentsPt"             , "constituentsPt"              , 200, 0, 10);
  TH1D *constituentsDptFromPrevious = new TH1D("constituentsDptFromPrevious", "constituentsDptFromPrevious" , 200, -4, 0);
  
  TH1D *constituentsSumDr           = new TH1D("constituentsSumDr" , "constituentsSumDr"  , 100, 0, 100);
  TH1D *constituentsSumDrPrevious   = new TH1D("constituentsSumDrPrevious" , "constituentsSumDrPrevious"  , 100, 0, 100);
  
  TH1D *constituentsAvgDr           = new TH1D("constituentsAvgDr" , "constituentsAvgDr"  , 100, 0.1, 0.6);
  TH1D *constituentsAvgDrPrevious   = new TH1D("constituentsAvgDrPrevious" , "constituentsAvgDrPrevious"  , 100, 0.2, 0.8);
  
  for(auto event : events){
    eventHists["MET"]->Fill(event->MET);
    eventHists["METeta"]->Fill(event->METeta);
    eventHists["METphi"]->Fill(event->METphi);
    eventHists["MT"]->Fill(event->MT);
    eventHists["Mjj"]->Fill(event->Mjj);
    
    for(auto jet : event->jets){
      jetHists["eta"]->Fill(jet->eta);
      jetHists["phi"]->Fill(jet->phi);
      jetHists["pt"]->Fill(jet->pt);
      jetHists["mass"]->Fill(jet->mass);
      jetHists["chargedFraction"]->Fill(jet->chargedFraction);
      jetHists["PTD"]->Fill(jet->PTD);
      jetHists["axis2"]->Fill(jet->axis2);
      jetHists["flavor"]->Fill(jet->flavor);
      jetHists["energy"]->Fill(jet->energy);
      
      for(int iEFP=0; iEFP<13; iEFP++){
        EFPhists[iEFP]->Fill(jet->EFPs[iEFP]);
      }
      
      double X = jet->mass;
      double Xreco = jet->pt * sqrt(0.1*jet->EFPs[1]/2);

      efpVErification->Fill(X, Xreco);
      efpMAss->Fill(Xreco);
      
      shared_ptr<Constituent> previousConstituent = nullptr;
      
      TLorentzVector jetVector;
      jetVector.SetPtEtaPhiM(jet->pt, jet->eta, jet->phi, jet->mass);
      
      double sumDr = 0;
      double sumDrPrevious = 0;
      int nSumDr = 0;
      int nSumDrPrevious = 0;
      
      for(auto constituent : jet->constituents){
        if(constituent->isEmpty()) continue;
        
        TLorentzVector constituentVector;
        constituentVector.SetPtEtaPhiE(constituent->pt, constituent->eta, constituent->phi, constituent->energy);
        
        double dEta = jet->eta - constituent->eta;
        double dPhi = jetVector.DeltaPhi(constituentVector);
        double dR = jetVector.DeltaR(constituentVector);
        
        constituentsDetaDphi->Fill(dEta, dPhi);
        constituentsDr->Fill(dR);
        sumDr += dR;
        nSumDr++;
        
        constituentsDrCorrected->Fill(dR);
        
        constituentsPt->Fill(constituent->pt);
        
        if(previousConstituent){
          TLorentzVector previousConstituentVector;
          previousConstituentVector.SetPtEtaPhiE(previousConstituent->pt, previousConstituent->eta,
                                                 previousConstituent->phi, previousConstituent->energy);
          
          double dPtPrevious  = constituent->pt - previousConstituent->pt;
          double dRprevious = constituentVector.DeltaR(previousConstituentVector);
          
          constituentsDrFromPrevious->Fill(dRprevious);
          sumDrPrevious += dRprevious;
          nSumDrPrevious++;
          
          constituentsDptFromPrevious->Fill(dPtPrevious);
        }
        
        previousConstituent = constituent;
      }
      
      constituentsSumDr->Fill(sumDr);
      constituentsSumDrPrevious->Fill(sumDrPrevious);
      
      constituentsAvgDr->Fill(sumDr/nSumDr);
      constituentsAvgDrPrevious->Fill(sumDrPrevious/nSumDrPrevious);
      
    }
    
    double lastValue = 0;
    
    for(int iBin=0; iBin<constituentsDrCorrected->GetNbinsX(); iBin++){
      double binContent = constituentsDrCorrected->GetBinContent(iBin);
      double binCenter = constituentsDrCorrected->GetXaxis()->GetBinCenter(iBin);
      
      double expectedContent = 2*TMath::Pi()*binCenter;
      constituentsDrCorrected->SetBinContent(iBin, binContent - expectedContent);
      
      if(binCenter < 0.8) lastValue = binContent - expectedContent;
    }
    
    for(int iBin=0; iBin<constituentsDrCorrected->GetNbinsX(); iBin++){
      double binContent = constituentsDrCorrected->GetBinContent(iBin);
      double binCenter = constituentsDrCorrected->GetXaxis()->GetBinCenter(iBin);
      constituentsDrCorrected->SetBinContent(iBin, binCenter < 0.8 ? binContent - lastValue : 0.0);
    }
    
    
    
    if(event->jets.size() != 2){
      cout<<"WARNING -- expected 2 jets in an event, but "<<event->jets.size()<<" were found!"<<endl;
      continue;
    }
    
    double dijetMass = event->jets[0]->mass + event->jets[1]->mass;
    jetsMasses->Fill(event->Mjj, dijetMass);
  }
  
  
  TCanvas *canvasEvents = new TCanvas("Events", "Events", 1000, 1500);
  canvasEvents->Divide(2, 3);
  
  int iPad=1;
  for(auto &[name, hist] : eventHists){
    canvasEvents->cd(iPad++);
    hist->DrawNormalized();
  }
  
  TCanvas *canvasJets = new TCanvas("Jets", "Jets", 2000, 1500);
  canvasJets->Divide(4, 3);
  
  iPad=1;
  canvasJets->cd(iPad++);
  efpVErification->Draw("colz");
  
  for(auto &[name, hist] : jetHists){
    canvasJets->cd(iPad++);
    hist->DrawNormalized();
    
    if(name=="mass") efpMAss->Draw("same");
  }
  
  TCanvas *canvasEFPs = new TCanvas("EFPs", "EFPs", 2000, 1500);
  canvasEFPs->Divide(4, 3);
  
  iPad=1;
  for(auto &[iEFP, hist] : EFPhists){
    canvasEFPs->cd(iPad++);
    hist->DrawNormalized();
  }
  
  gStyle->SetOptStat(0);
  
  TCanvas *canvasConstituents = new TCanvas("Constituents", "Constituents", 2000, 2000);
  canvasConstituents->Divide(4, 3);
  
  canvasConstituents->cd(1);
  constituentsDetaDphi->Draw("colz");
  constituentsDetaDphi->GetZaxis()->SetRangeUser(0, 200);
  
  canvasConstituents->cd(2);
  constituentsDr->Draw();
  
  canvasConstituents->cd(3);
  constituentsDrCorrected->Draw();
  
  canvasConstituents->cd(4);
  gPad->SetLogy();
  constituentsPt->Draw();
  
  canvasConstituents->cd(5);
  gPad->SetLogy();
  constituentsDptFromPrevious->Draw();
  
  canvasConstituents->cd(6);
  gPad->SetLogy();
  constituentsDrFromPrevious->Draw();
  
  canvasConstituents->cd(7);
//  gPad->SetLogy();
  constituentsSumDr->Draw();
  
  canvasConstituents->cd(8);
//  gPad->SetLogy();
  constituentsSumDrPrevious->Draw();
  
  canvasConstituents->cd(9);
  constituentsAvgDr->Draw();
  
  canvasConstituents->cd(10);
  constituentsAvgDrPrevious->Draw();
  
  
  canvasEvents->Update();
  canvasJets->Update();
  canvasEFPs->Update();
  canvasConstituents->Update();
  
  
  TFile *outfile = new TFile(outputPath.c_str(), "recreate");
  outfile->cd();
  
  constituentsDetaDphi->Write();
  constituentsDr->Write();
  constituentsDrCorrected->Write();
  constituentsPt->Write();
  constituentsDptFromPrevious->Write();
  constituentsDrFromPrevious->Write();
  constituentsSumDr->Write();
  constituentsSumDrPrevious->Write();
  constituentsAvgDr->Write();
  constituentsAvgDrPrevious->Write();
  
  jetHists["pt"]->Write();
  jetHists["mass"]->Write();
  jetHists["PTD"]->Write();
  jetHists["axis2"]->Write();
  EFPhists[1]->Write();
  
  outfile->Close();
  
  theApp.Run();
  return 0;
}
