from enum import Enum


class InputTypes(Enum):
    Delphes = 0,
    nanoAOD = 1,
    PFnanoAOD102X = 2,
    PFnanoAOD106X = 3,
    scoutingAtHlt = 4

class DataProcessor:
    def __init__(self, tree, input_type):
        """
        Creates DataProcessor objects which knows names of branches for different input types.
        Pre-loads all branches for later use.
        """
        
        if input_type not in InputTypes:
            print("\n\nERROR -- DataProcessor: unknown input type: ", input_type, "\n\n")
            exit(0)
        
        self.variables = {
            InputTypes.Delphes: {
                # number of objects
                "N_jets": "Jet_size",
                "N_fat_jets": "FatJet_size",
                "N_tracks": "EFlowTrack_size",
                "N_neutral_hadrons": "EFlowNeutralHadron_size",
                "N_photons": "Photon_size",
                # event features
                "MET_pt": "MissingET/MissingET.MET",
                "MET_eta": "MissingET/MissingET.Eta",
                "MET_phi": "MissingET/MissingET.Phi",
                # jet features
                "Jet_eta": "Jet/Jet.Eta",
                "Jet_phi": "Jet/Jet.Phi",
                "Jet_pt": "Jet/Jet.PT",
                "Jet_mass": "Jet/Jet.Mass",
                "Jet_nCharged": "Jet/Jet.NCharged",
                "Jet_nNeutral": "Jet/Jet.NNeutrals",
                "Jet_flavor": "Jet/Jet.Flavor",
                # fat jet features
                "FatJet_eta": "FatJet/FatJet.Eta",
                "FatJet_phi": "FatJet/FatJet.Phi",
                "FatJet_pt": "FatJet/FatJet.PT",
                "FatJet_mass": "FatJet/FatJet.Mass",
                "FatJet_nCharged": "FatJet/FatJet.NCharged",
                "FatJet_nNeutral": "FatJet/FatJet.NNeutrals",
                "FatJet_flavor": "FatJet/FatJet.Flavor",
                # tracks
                "Track_eta": "EFlowTrack/EFlowTrack.Eta",
                "Track_phi": "EFlowTrack/EFlowTrack.Phi",
                "Track_pt": "EFlowTrack/EFlowTrack.PT",
                # neutral hadrons
                "Neutral_eta": "EFlowNeutralHadron/EFlowNeutralHadron.Eta",
                "Neutral_phi": "EFlowNeutralHadron/EFlowNeutralHadron.Phi",
                "Neutral_pt": "EFlowNeutralHadron/EFlowNeutralHadron.ET",
                # photons
                "Photon_eta": "Photon/Photon.Eta",
                "Photon_phi": "Photon/Photon.Phi",
                "Photon_pt": "Photon/Photon.PT",
            },
            InputTypes.nanoAOD: {
                # number of objects
                "N_jets": "nJet",
                "N_fat_jets": "nFatJet",
                "N_photons": "nPhoton",
                # event features
                "MET_pt": "MET_pt",
                "MET_phi": "MET_phi",
                "Gen_weight": "genWeight",
                # jet features
                "Jet_eta": "Jet_eta",
                "Jet_phi": "Jet_phi",
                "Jet_pt": "Jet_pt",
                "Jet_mass": "Jet_mass",
                "Jet_chHEF": "Jet_chHEF",
                "Jet_neHEF": "Jet_neHEF",
                # photons
                "Photon_eta": "Photon_eta",
                "Photon_phi": "Photon_phi",
                "Photon_pt": "Photon_pt",
                "Photon_mass": "Photon_mass",
            },
            InputTypes.PFnanoAOD102X: {
                # number of objects
                "N_jets": "nJet",
                "N_fat_jets": "nFatJet",
                "N_tracks_AK4": "nJetPFCands",
                "N_tracks_AK8": "nFatJetPFCands",
                "N_photons": "nPhoton",
                # event features
                "MET_pt": "MET_pt",
                "MET_phi": "MET_phi",
                "Gen_weight": "genWeight",
                # jet features
                "Jet_eta": "Jet_eta",
                "Jet_phi": "Jet_phi",
                "Jet_pt": "Jet_pt",
                "Jet_mass": "Jet_mass",
                "Jet_chHEF": "Jet_chHEF",
                "Jet_neHEF": "Jet_neHEF",
                # fat jet features
                "FatJet_eta": "FatJet_eta",
                "FatJet_phi": "FatJet_phi",
                "FatJet_pt": "FatJet_pt",
                "FatJet_mass": "FatJet_mass",
                # tracks for jets
                "Track_eta_AK4": "JetPFCands_eta",
                "Track_phi_AK4": "JetPFCands_phi",
                "Track_pt_AK4": "JetPFCands_pt",
                "Track_mass_AK4": "JetPFCands_mass",
                "Track_jet_index_AK4": "JetPFCands_jetIdx",
                "Track_pid_AK4": "JetPFCands_pdgId",
                # tracks for fat jets
                "Track_eta_AK8": "FatJetPFCands_eta",
                "Track_phi_AK8": "FatJetPFCands_phi",
                "Track_pt_AK8": "FatJetPFCands_pt",
                "Track_mass_AK8": "FatJetPFCands_mass",
                "Track_jet_index_AK8": "FatJetPFCands_jetIdx",
                "Track_pid_AK8": "FatJetPFCands_pdgId",
                # photons
                "Photon_eta": "Photon_eta",
                "Photon_phi": "Photon_phi",
                "Photon_pt": "Photon_pt",
                "Photon_mass": "Photon_mass",
            },
            InputTypes.PFnanoAOD106X: {
                # number of objects
                "N_jets": "nJet",
                "N_fat_jets": "nFatJet",
                "N_tracks": "nJetPFCands",
                "N_photons": "nPhoton",
                # event features
                "MET_pt": "MET_pt",
                "MET_phi": "MET_phi",
                "Gen_weight": "genWeight",
                # jet features
                "Jet_eta": "Jet_eta",
                "Jet_phi": "Jet_phi",
                "Jet_pt": "Jet_pt",
                "Jet_mass": "Jet_mass",
                "Jet_chHEF": "Jet_chHEF",
                "Jet_neHEF": "Jet_neHEF",
                # fat jet features
                "FatJet_eta": "FatJet_eta",
                "FatJet_phi": "FatJet_phi",
                "FatJet_pt": "FatJet_pt",
                "FatJet_mass": "FatJet_mass",
                # tracks
                "Track_eta": "JetPFCands_eta",
                "Track_phi": "JetPFCands_phi",
                "Track_pt": "JetPFCands_pt",
                "Track_mass": "JetPFCands_mass",
                "Track_jet_index_AK4": "JetPFCandsAK4_jetIdx",
                "Track_cand_index_AK4": "JetPFCandsAK4_candIdx",
                "Track_jet_index_AK8": "JetPFCandsAK8_jetIdx",
                "Track_cand_index_AK8": "JetPFCandsAK8_candIdx",
                "Track_pid": "JetPFCands_pdgId",
                # photons
                "Photon_eta": "Photon_eta",
                "Photon_phi": "Photon_phi",
                "Photon_pt": "Photon_pt",
                "Photon_mass": "Photon_mass",
            },
	    InputTypes.scoutingAtHlt:{
                # number of objects
                #"N_jets": "Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018./Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.pt_",
                #"N_tracks": "Run3ScoutingTracks_hltScoutingTrackPacker__HLT2018./Run3ScoutingTracks_hltScoutingTrackPacker__HLT2018.obj.pt_",
                #"N_photons": "Run3ScoutingPhotons_hltScoutingEgammaPacker__HLT2018./Run3ScoutingPhotons_hltScoutingEgammaPacker__HLT2018.obj.pt_",
                # event features
                "MET_pt": "double_hltScoutingPFPacker_pfMetPt_HLT2018./double_hltScoutingPFPacker_pfMetPt_HLT2018.obj",
                "MET_phi": "double_hltScoutingPFPacker_pfMetPhi_HLT2018./double_hltScoutingPFPacker_pfMetPhi_HLT2018.obj",
                # jet features
                "Jet_eta": "Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018./Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj/Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.eta_",
                "Jet_phi": "Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018./Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj/Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.phi_",
                "Jet_pt": "Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018./Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj/Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.pt_",
                "Jet_mass": "Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018./Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj/Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.m_",
                "Jet_nCharged": "Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018./Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj/Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.chargedHadronMultiplicity_",# ! only take hadrons into account !
                "Jet_nNeutral": "Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018./Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj/Run3ScoutingPFJets_hltScoutingPFPacker__HLT2018.obj.neutralHadronMultiplicity_",# ! only take hadrons into account !
                # tracks
                "Track_eta": "Run3ScoutingTracks_hltScoutingTrackPacker__HLT2018./Run3ScoutingTracks_hltScoutingTrackPacker__HLT2018.obj/Run3ScoutingTracks_hltScoutingTrackPacker__HLT2018.obj.tk_eta_",
                "Track_phi": "Run3ScoutingTracks_hltScoutingTrackPacker__HLT2018./Run3ScoutingTracks_hltScoutingTrackPacker__HLT2018.obj/Run3ScoutingTracks_hltScoutingTrackPacker__HLT2018.obj.tk_phi_",
                "Track_pt": "Run3ScoutingTracks_hltScoutingTrackPacker__HLT2018./Run3ScoutingTracks_hltScoutingTrackPacker__HLT2018.obj/Run3ScoutingTracks_hltScoutingTrackPacker__HLT2018.obj.tk_pt_",
                # photons
                "Photon_eta": "Run3ScoutingPhotons_hltScoutingEgammaPacker__HLT2018./Run3ScoutingPhotons_hltScoutingEgammaPacker__HLT2018.obj/Run3ScoutingPhotons_hltScoutingEgammaPacker__HLT2018.obj.eta_",
                "Photon_phi": "Run3ScoutingPhotons_hltScoutingEgammaPacker__HLT2018./Run3ScoutingPhotons_hltScoutingEgammaPacker__HLT2018.obj/Run3ScoutingPhotons_hltScoutingEgammaPacker__HLT2018.obj.phi_",
                "Photon_pt": "Run3ScoutingPhotons_hltScoutingEgammaPacker__HLT2018./Run3ScoutingPhotons_hltScoutingEgammaPacker__HLT2018.obj/Run3ScoutingPhotons_hltScoutingEgammaPacker__HLT2018.obj.pt_",	
	    }
        }
        
        # pre-load all branches for this tree to avoid calling this for every event/track/jet
        self.branches = {}
        
        print("Keys found in the tree:", tree.keys())
         
        for key, value in self.variables[input_type].items():
            if value in tree.keys():
                self.branches[key] = tree[value].array()
        
    def get_value_from_tree(self, variable, i_event=None, i_entry=None):
        """
        Returns value of given variable for given event. If event contains an array of such variable (e.g. jets pt),
        i_entry must be also specified.
        """
        if variable not in self.branches.keys():
            return None
        
        if i_entry is None:
            return self.branches[variable][i_event]
        else:
            return self.branches[variable][i_event][i_entry]
        
    def get_array_n_dimensions(self, variable):
        """
        Returns number of dimensions of the tree leaf for given variable (1 is a number, 2 is a vector etc.)
        """
        
        if variable not in self.branches.keys():
            return None
    
        return self.branches[variable].ndim
