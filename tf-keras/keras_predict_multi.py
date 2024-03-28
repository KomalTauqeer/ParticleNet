import os
import sys
sys.path.append("preprocessing/")
import optparse
import numpy as np
from numpy import savetxt
import utilis
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#plt.rcParams.update({'font.size': 12})
plt.rcParams["mathtext.fontset"] = "cm"
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import roc_curve,RocCurveDisplay,auc
from sklearn.metrics import classification_report
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
#from scipy import interp
from itertools import cycle
from load_datasets import Dataset
import rootIO
import meta_data

parser = optparse.OptionParser()
parser.add_option("--year", dest="year", default= "UL18")
parser.add_option("--region", dest="region", default= 'VBSSR')
parser.add_option("--do_eval" , "--do_eval", action="store_true", dest = "do_eval", help = "predict on MC samples", default = False)
parser.add_option("--do_eval_data" , "--do_eval_data", action="store_true", dest = "do_eval_data", help = "predict on data samples", default = False)
parser.add_option("--do_eval_sys" , "--do_eval_sys", action="store_true", dest = "do_eval_sys", help = "predict on sys samples", default = False)
(options,args) = parser.parse_args()
year = options.year
region = options.region

samples = ["TT", "ST", "WJet", "ssWW", "osWW", "WZ", "ZZ", "QCDVV"]
data_samples = ["Data_muon", "Data_electron"]

treename = "AnalysisTree"

inputfilepath = { 
                   'VBSSR': {
                             "UL16preVFP": "/ceph/ktauqeer/ULNtuples/UL16preVFP/VBSSR/" ,
                             "UL16postVFP": "/ceph/ktauqeer/ULNtuples/UL16postVFP/VBSSR/",
                             "UL17": "/ceph/ktauqeer/ULNtuples/UL17/VBSSR/",
                             "UL18": "/ceph/ktauqeer/ULNtuples/UL18/VBSSR/",
                            },
                }

inputfilename = { 

                  #'VBSSR': {
                  #       "TT": "VBSSR_TTToSemiLeptonic_ABC_dnn_sr1_0p6.root",
                  #       "WJet": "VBSSR_WJetsToLNu_combined_ABC_dnn_sr1_0p6.root",
                  #       "ST": "VBSSR_ST_combined_ABC_dnn_sr1_0p6.root",
                  #       "ssWW": "VBSSR_ssWW_combined_ABC_dnn_sr1_0p6.root",
                  #       "osWW": "VBSSR_osWW_combined_ABC_dnn_sr1_0p6.root",
                  #       "WZ": "VBSSR_WZ_combined_ABC_dnn_sr1_0p6.root",
                  #       "ZZ": "VBSSR_ZZ_ABC_dnn_sr1_0p6.root",
                  #       "QCDVV": "VBSSR_QCDVV_combined_ABC_dnn_sr1_0p6.root",
                  #       },
                
                   'VBSSR': {
                         "TT": "VBSSR_TTToSemiLeptonic_16032024_ABC_dnn_sr1_0p6.root",
                         "WJet": "VBSSR_WJetsToLNu_combined_16032024_ABC_dnn_sr1_0p6.root",
                         "ST": "VBSSR_ST_combined_16032024_ABC_dnn_sr1_0p6.root",
                         "ssWW": "VBSSR_ssWW_combined_16032024_ABC_dnn_sr1_0p6.root",
                         "osWW": "VBSSR_osWW_combined_16032024_ABC_dnn_sr1_0p6.root",
                         "WZ": "VBSSR_WZ_combined_16032024_ABC_dnn_sr1_0p6.root",
                         "ZZ": "VBSSR_ZZ_16032024_ABC_dnn_sr1_0p6.root",
                         "QCDVV": "VBSSR_QCDVV_combined_16032024_ABC_dnn_sr1_0p6.root",
                         },
                }

datafilename =  { 'VBSSR': {
                         "Data_muon": "VBSSR_SingleMuon_combined_dnn_sr1_0p6.root",
                         "Data_electron": "VBSSR_SingleElectron_combined_dnn_sr1_0p6.root",
                         }
                }

datafilename_UL18 =  { 'VBSSR': {
                         "Data_muon": "VBSSR_SingleMuon_combined_16032024_dnn_sr1_0p6.root",
                         "Data_electron": "VBSSR_EGamma_combined_16032024_dnn_sr1_0p6.root",
                         }
                }

inputfilename_sys = {
                       'VBSSR': {
                         "TT": "VBSSR_TTToSemiLeptonic_dnn_sr1_0p6.root",
                         "WJet": "VBSSR_WJetsToLNu_combined_dnn_sr1_0p6.root",
                         "ST": "VBSSR_ST_combined_dnn_sr1_0p6.root",
                         "ssWW": "VBSSR_ssWW_combined_dnn_sr1_0p6.root",
                         "osWW": "VBSSR_osWW_combined_dnn_sr1_0p6.root",
                         "WZ": "VBSSR_WZ_combined_dnn_sr1_0p6.root",
                         "ZZ": "VBSSR_ZZ_dnn_sr1_0p6.root",
                         "QCDVV": "VBSSR_QCDVV_combined_dnn_sr1_0p6.root",

                        },
                   }


def main():

    #Load model
    #model = keras.models.load_model("successful_multitrain_results/Oct23_lrsch_1e-4/PNL_checkpoints_lrsch_1e-4/particle_net_lite_model.029.h5")
    model = keras.models.load_model("successful_genmatchedZ_multitrain/model_checkpoints/particle_net_lite_model.030.h5")

    if options.do_eval_data:
        for sample in data_samples:
            eval_dataset = Dataset('preprocessing/converted/data_eval_sets/{}_{}_{}_0.awkd'.format(region, sample, year), data_format='channel_last')
            PN_output= (model.predict(eval_dataset.X))
            print ("Output for {}: ".format(sample))
            print (PN_output)
            predicted_class = np.array([np.argmax(PN_output, axis=1)]).T
            print (predicted_class.flatten())
            nrows, ncolumns = (np.shape(PN_output))
            print (np.shape(predicted_class))
            predicted_probabilites = []
            for row in range(nrows):
                predicted_probabilites.append(PN_output[row][predicted_class[row]])
            print (np.array(predicted_probabilites).flatten())
            print (np.shape(predicted_probabilites))
            print ()
            if year!= "UL18": file_path = inputfilepath[region][year] + datafilename[region][sample]
            else: file_path = inputfilepath[region][year] + datafilename_UL18[region][sample]
            rootIO.add_branches(file_path, treename, 'jetchargetagger_prob', 'F', np.array(predicted_probabilites).flatten(), 'jetchargetagger_ind', 'F', predicted_class.flatten())
            print ("Branches added to the root file {}".format(file_path))

    elif options.do_eval:
        for sample in samples:
            eval_dataset = Dataset('preprocessing/converted/eval_sets/{}_{}_{}_0.awkd'.format(region, sample, year), data_format='channel_last')
            PN_output= (model.predict(eval_dataset.X))
            print ("Output for {}: ".format(sample))
            print (PN_output)
            predicted_class = np.array([np.argmax(PN_output, axis=1)]).T
            print (predicted_class.flatten())
            nrows, ncolumns = (np.shape(PN_output))
            print (np.shape(predicted_class))
            predicted_probabilites = []
            for row in range(nrows):
                predicted_probabilites.append(PN_output[row][predicted_class[row]])
            print (np.array(predicted_probabilites).flatten())
            print (np.shape(predicted_probabilites))
            print ()
            file_path = inputfilepath[region][year] + inputfilename[region][sample]
            rootIO.add_branches(file_path, treename, 'jetchargetagger_prob', 'F', np.array(predicted_probabilites).flatten(), 'jetchargetagger_ind', 'F', predicted_class.flatten())
            print ("Branches added to the root file {}".format(file_path))

    elif options.do_eval_sys:
        for sample in samples:
            for systype in ["jec", "jer"]:
                for sysdir in ["up", "down"]:
                    eval_dataset = Dataset('preprocessing/converted/sys/{}_{}_{}_{}_{}_0.awkd'.format(region, sample, year, systype, sysdir), data_format='channel_last')
                    PN_output= (model.predict(eval_dataset.X))
                    print ("Output for {}: ".format(sample))
                    print (PN_output)
                    predicted_class = np.array([np.argmax(PN_output, axis=1)]).T
                    print (predicted_class.flatten())
                    nrows, ncolumns = (np.shape(PN_output))
                    print (np.shape(predicted_class))
                    predicted_probabilites = []
                    for row in range(nrows):
                        predicted_probabilites.append(PN_output[row][predicted_class[row]])
                    print (np.array(predicted_probabilites).flatten())
                    print (np.shape(predicted_probabilites))
                    print ()
                    file_path = meta_data.inputfilepath['VBSSR'][year]+systype+'/'+sysdir+'/' 
                    filename = meta_data.inputfilename_sys['VBSSR'][sample][:-len('_dnn_sr1_0p6.root')]+'_'+systype+'_'+sysdir+'_dnn_sr1_0p6.root'
                    rootIO.add_branches(file_path+filename, treename, 'jetchargetagger_prob', 'F', np.array(predicted_probabilites).flatten(), 'jetchargetagger_ind', 'F', predicted_class.flatten())
                    print ("Branches added to the root file {}".format(file_path))

    else: 
        print ("No valid eval option selected!")

if __name__ == '__main__':
    main()
