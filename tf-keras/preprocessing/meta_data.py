variables = ["PF_Px", "PF_Py", "PF_Pz", "PF_E", "PF_q"]
labels = ["lep_charge"]
treename = "AnalysisTree"
inputfilepath = {  'TTCR': {
                       "UL16preVFP": "/ceph/ktauqeer/ULNtuples/UL16preVFP/TTCR/" ,
                       "UL16postVFP": "/ceph/ktauqeer/ULNtuples/UL16postVFP/TTCR/",
                       "UL17": "/ceph/ktauqeer/ULNtuples/UL17/TTCR/",
                       "UL18": "/ceph/ktauqeer/ULNtuples/UL18/TTCR/",
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
                  'VBSSR': {
                            "ssWW": "VBSSR_ssWW_combined.root",
                            "osWW": "VBSSR_osWW_combined.root",
                            "SingleMuon": "VBSSR_SingleMuon_combined.root",
                            "SingleElectron": "VBSSR_SingleElectron_combined.root",
                           },
                }
   


