variables = ["PF_Px", "PF_Py", "PF_Pz", "PF_E", "PF_q"] #Puppi weights are applied already
weights = ["event_weight"]
labels = ["lep_charge"]
treename = "AnalysisTree"

inputfilepath = {  'TTCR': {
                       "UL16preVFP": "/ceph/ktauqeer/ULNtuples/UL16preVFP/TTCR/" ,
                       "UL16postVFP": "/ceph/ktauqeer/ULNtuples/UL16postVFP/TTCR/",
                       "UL17": "/ceph/ktauqeer/ULNtuples/UL17/TTCR/",
                       "UL18": "/ceph/ktauqeer/ULNtuples/UL18/TTCR/",
                      },
                   'ZJetsCR': {
                             "UL16preVFP": "/ceph/ktauqeer/ULNtuples/UL16preVFP/ZJetsCR/" ,
                             "UL16postVFP": "/ceph/ktauqeer/ULNtuples/UL16postVFP/ZJetsCR/",
                             "UL17": "/ceph/ktauqeer/ULNtuples/UL17/ZJetsCR/",
                             "UL18": "/ceph/ktauqeer/ULNtuples/UL18/ZJetsCR/",
                             },
                 }


inputfilename = { 'TTCR': {
                      "TT": "TTCR_TTToSemiLeptonic.root",
                      "WJet": "TTCR_WJetsToLNu_combined.root",
                      "ST": "TTCR_ST_combined.root",
                      "QCD": "TTCR_QCD_combined.root",

                     },
                  'ZJetsCR': {
                             "ZJets": "ZJetsCR_ZJets_{}_genmatched_reduced.root",
                             },

                }

datafilename =  { 'TTCR': {"Data_muon": "TTCR_SingleMuon_combined.root",
                           "Data_electron": "TTCR_SingleElectron_combined.root",
                         },

                }

datafilename_UL18 =  { 'TTCR': {"Data_muon": "TTCR_SingleMuon_combined.root",
                           "Data_electron": "TTCR_EGamma_combined.root",
                         },

                }

   


