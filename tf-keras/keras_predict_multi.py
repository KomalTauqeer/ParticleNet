import os
import sys
import optparse
import numpy as np
from numpy import savetxt
import awkward0
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

parser = optparse.OptionParser()
parser.add_option("--year", dest="year", default= "UL18")
parser.add_option("--region", dest="region", default= 'VBSSR')
(options,args) = parser.parse_args()
year = options.year
region = options.region
samples = ["TT", "ST", "WJet", "ssWW", "osWW", "WZ", "ZZ", "QCDVV"]

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

                  'VBSSR': {
                         "TT": "VBSSR_TTToSemiLeptonic_ABC_dnn_sr1.root",
                         "WJet": "VBSSR_WJetsToLNu_combined_ABC_dnn_sr1.root",
                         "ST": "VBSSR_ST_combined_ABC_dnn_sr1.root",
                         "ssWW": "VBSSR_ssWW_combined_ABC_dnn_sr1.root",
                         "osWW": "VBSSR_osWW_combined_ABC_dnn_sr1.root",
                         "WZ": "VBSSR_WZ_combined_ABC_dnn_sr1.root",
                         "ZZ": "VBSSR_ZZ_ABC_dnn_sr1.root",
                         "QCDVV": "VBSSR_QCDVV_combined_ABC_dnn_sr1.root",
                         },
                }

def main():

    #Load model
    model = keras.models.load_model("successful_multitrain_results/Oct23_lrsch_1e-4/PNL_checkpoints_lrsch_1e-4/particle_net_lite_model.029.h5")
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


if __name__ == '__main__':
    main()
