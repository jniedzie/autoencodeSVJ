#include "Helpers.hpp"

#include "EventProcessor.hpp"


/*
 For this to work you'll need to install HDF5 together with the C++ API. On macOS, you can do that with brew:
 
 `brew install hdf5`
 
 Then, find where the HDF5 C++ API header and libs are located and build executable with:
 
 g++ h5parser.cpp -o h5parser `root-config --libs` `root-config --cflags` -I/usr/local/Cellar/hdf5/1.12.0_1/include/ -L/usr/local/Cellar/hdf5/1.12.0_1/lib/ -lhdf5_cpp -lhdf5
 
 */

//const string inputPath = "/Users/Jeremi/Documents/Physics/ETH/data/training_data/qcd/base_3/data_0_data.h5";
const string inputPath = "/Users/Jeremi/Documents/Physics/ETH/data/backgrounds/qcd/h5_no_lepton_veto_fat_jets_dr0p8_withConstituents/base_3/QCD_part_1.h5";



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
  
  for(auto event : events) event->print();
  
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
  
  
  canvasEvents->Update();
  canvasJets->Update();
  canvasEFPs->Update();
  
  theApp.Run();
  return 0;
}
