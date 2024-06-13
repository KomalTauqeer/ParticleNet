variables = ["PF_Px", "PF_Py", "PF_Pz", "PF_E", "PF_q", "PF_puppiweight"]
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
                             "UL18": "/ceph/ktauqeer/JetChargeTaggerNtuples/UL18/ZJetsCR/",
                             },
                   'VBSSR': {
                             "UL16preVFP": "/ceph/ktauqeer/ULNtuples/UL16preVFP/VBSSR/" ,
                             "UL16postVFP": "/ceph/ktauqeer/ULNtuples/UL16postVFP/VBSSR/",
                             "UL17": "/ceph/ktauqeer/ULNtuples/UL17/VBSSR/",
                             "UL18": "/ceph/ktauqeer/ULNtuples/UL18/VBSSR/",
                            },
                 }

inputfilename = { 'TTCR': {
                      "TT": "TTCR_TTToSemiLeptonic.root",
                     },
                  'ZJetsCR': {
                          "ZJets": "ZJetsCR_ZJets_HTcombined_mod2.root",
                          },
                  'VBSSR': {
                            "ssWW": "VBSSR_ssWW_combined.root",
                            "osWW": "VBSSR_osWW_combined.root",
                           },
                }

datafilename =  { 'TTCR': {"Data_singlemuon": "TTCR_SingleMuon_combined.root",
                           "Data_singleelectron": "TTCR_SingleElectron_combined.root",
                         },

                  'VBSSR': {
                         "Data_singlemuon": "VBSSR_SingleMuon_combined_dnn_sr1_0p6.root",
                         "Data_singleelectron": "VBSSR_SingleElectron_combined_dnn_sr1_0p6.root",
                         },
                }

datafilename_UL18 =  { 'TTCR': {"Data_singlemuon": "TTCR_SingleMuon_combined.root",
                           "Data_singleelectron": "TTCR_EGamma_combined.root",
                         },

                  'VBSSR': {
                         "Data_singlemuon": "VBSSR_SingleMuon_combined_dnn_sr1_0p6.root",
                         "Data_singleelectron": "VBSSR_EGamma_combined_dnn_sr1_0p6.root",
                         },
                }

   


