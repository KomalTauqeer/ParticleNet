variables = ["PF_Px", "PF_Py", "PF_Pz", "PF_E", "PF_q"]
labels = ['lep_charge']
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
                   'VBSSR': {
                             "UL16preVFP": "/ceph/ktauqeer/ULNtuples/UL16preVFP/VBSSR/" ,
                             "UL16postVFP": "/ceph/ktauqeer/ULNtuples/UL16postVFP/VBSSR/",
                             "UL17": "/ceph/ktauqeer/ULNtuples/UL17/VBSSR/",
                             "UL18": "/ceph/ktauqeer/ULNtuples/UL18/VBSSR/",
                            },
                 }

inputfilename = { 'TTCR': {
                        "TT": "TTCR_TTToSemiLeptonic.root",
                        "SingleMuon": "TTCR_SingleMuon_combined.root",
                        "SingleElectron": "TTCR_SingleElectron_combined.root",
                        },
                  'ZJetsCR': {
                          "ZJets": "ZJetsCR_ZJets_UL18_genmatchedZ_0To645000.root",

                           },
                  'VBSSR': {
                         "TT": "VBSSR_TTToSemiLeptonic.root",
                         "WJet": "VBSSR_WJetsToLNu_combined.root",
                         "ST": "VBSSR_ST_combined.root",
                         "ssWW": "VBSSR_ssWW_combined.root",
                         "osWW": "VBSSR_osWW_combined.root",
                         "WZ": "VBSSR_WZ_combined.root",
                         "ZZ": "VBSSR_ZZ.root",
                         "QCDVV": "VBSSR_QCDVV_combined.root",
                         },

                }

