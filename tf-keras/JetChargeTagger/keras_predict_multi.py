import os
import sys
sys.path.append("preprocessing/")
import numpy as np
import optparse
import tensorflow as tf
from tensorflow import keras
from load_datasets import Dataset
from array import array
from ROOT import *
import rootIO
from constants import *

parser = optparse.OptionParser()
parser.add_option("--year", dest="year", default= "UL18")
parser.add_option("--region", dest="region", default= 'TTCR')
parser.add_option("--do_sys" , "--do_sys", action="store_true", dest = "do_sys", help = "predict jec and jer files", default = False)
parser.add_option("--do_eval" , "--do_eval", action="store_true", dest = "do_eval", help = "predict on MC + data samples", default = False)
(options,args) = parser.parse_args()
year = options.year
region = options.region

def load_model():

    modelnumber = {"UL16preVFP": 30, "UL16postVFP": 30, "UL17": 30, "UL18": 29}
    modelpath = "/work/ktauqeer/ParticleNet/tf-keras/preprocessing/JetChargeTagger/ternary_training/{}/model_checkpoints/particle_net_lite_model.0{}.h5".format(year, modelnumber[year])
    model = keras.models.load_model(modelpath)

    return model

def predict_testset():
    
    #Load model
    model = load_model()
 
    eval_path = "preprocessing/ternary_training/{y}/converted/WpWnZ_test_{y}_0.awkd".format(y=year)
    print ("********************* Evaluating {} *****************************".format(eval_path))
    eval_dataset = Dataset(eval_path, data_format='channel_last')
    PN_output= model.predict(eval_dataset.X)
    print (PN_output)
    true_class = np.array([np.argmax(eval_dataset.y, axis=1)]).T
    predicted_class = np.array([np.argmax(PN_output, axis=1)]).T
    nrows, ncolumns = np.shape(PN_output)
    predicted_probabilites = []
    for row in range (nrows):
        predicted_probabilites.append(PN_output[row][predicted_class[row]])

    ofile = TFile.Open('ternary_training/{}/WpWnZ_test.root'.format(year), "RECREATE")
    tree = TTree("AnalysisTree", "AnalysisTree")
    Wp = array('d', [0])
    Wn = array('d', [0])
    Z = array('d', [0])
    Ind = array('d', [0])
    trueInd = array('d', [0])
    tree.Branch('jetchargetagger_prob_nodeWp', Wp, 'jetchargetagger_prob_nodeWp/D')
    tree.Branch('jetchargetagger_prob_nodeWn', Wn, 'jetchargetagger_prob_nodeWn/D')
    tree.Branch('jetchargetagger_prob_nodeZ', Z, 'jetchargetagger_prob_nodeZ/D')
    tree.Branch('jetchargetagger_ind', Ind, 'jetchargetagger_ind/D')
    tree.Branch('jetchargetagger_true_ind', trueInd, 'jetchargetagger_true_ind/D')
    for itr in range(nrows):
        Wp[0] = PN_output[itr,0]
        Wn[0] = PN_output[itr,1]
        Z[0] = PN_output[itr,2]
        Ind[0] = predicted_class.flatten()[itr]
        trueInd[0] = true_class.flatten()[itr]
        tree.Fill()

    tree.Write()
    ofile.Write()
    ofile.Close()
    print ("Branches added to the root file {}".format(ofile))

def predict_mc():
    
    #Load model
    model = load_model()
 
    for sample in samples:
        eval_path = "preprocessing/ternary_training/{y}/Eval/converted/Eval_TTCR_{s}_{y}_0.awkd".format(y=year, s=sample)
        print ("********************* Evaluating {} *****************************".format(eval_path))
        eval_dataset = Dataset(eval_path, data_format='channel_last')
        #print (eval_dataset)
        PN_output= model.predict(eval_dataset.X)
        #print ("Output for {}: ".format(sample))
        #print (PN_output)
        predicted_class = np.array([np.argmax(PN_output, axis=1)]).T
        #print (predicted_class.flatten())
        nrows, ncolumns = np.shape(PN_output)
        #print (np.shape(predicted_class))
        predicted_probabilites = []
        for row in range (nrows):
            predicted_probabilites.append(PN_output[row][predicted_class[row]])
        #print (np.array(predicted_probabilites).flatten())
        #print (np.shape(predicted_probabilites))
        #print ()

        ofile_path = inputfilepath[region][year] + inputfilename[region][sample].rstrip('.root') + '_test.root'
        rootIO.add_fourbranches(ofile_path, treename, 'jetchargetagger_prob_nodeWp', 'F', np.array(PN_output[:,0]).flatten(), 'jetchargetagger_prob_nodeWn', 'F', np.array(PN_output[:,1]).flatten(),'jetchargetagger_prob_nodeZ', 'F', np.array(PN_output[:,2]).flatten(), 'jetchargetagger_ind', 'F', predicted_class.flatten())
        print ("Branches added to the root file {}".format(ofile_path))


def predict_data():

    #Load model
    model = load_model()

    for sample in data_samples:
        eval_path = "preprocessing/ternary_training/{y}/Data/converted/Eval_TTCR_{s}_{y}_0.awkd".format(y=year, s=sample)
        print ("********************* Evaluating {} *****************************".format(eval_path))
        eval_dataset = Dataset(eval_path, data_format='channel_last')
        PN_output= model.predict(eval_dataset.X)
        predicted_class = np.array([np.argmax(PN_output, axis=1)]).T
        nrows, ncolumns = np.shape(PN_output)
        predicted_probabilites = []
        for row in range (nrows):
            predicted_probabilites.append(PN_output[row][predicted_class[row]])

        if year == "UL18":
            ofile_path = inputfilepath[region][year] + datafilename_UL18[region][sample].rstrip('.root') + '_test.root'
        else:
            ofile_path = inputfilepath[region][year] + datafilename[region][sample].rstrip('.root') + '_test.root'

        rootIO.add_fourbranches(ofile_path, treename, 'jetchargetagger_prob_nodeWp', 'F', np.array(PN_output[:,0]).flatten(), 'jetchargetagger_prob_nodeWn', 'F', np.array(PN_output[:,1]).flatten(),'jetchargetagger_prob_nodeZ', 'F', np.array(PN_output[:,2]).flatten(), 'jetchargetagger_ind', 'F', predicted_class.flatten())
        print ("Branches added to the root file {}".format(ofile_path))

def predict_sys():

    #Load model
    model = load_model()
    for systype in ["jec", "jer"]:
        for sysdir in ["up", "down"]:
            for sample in samples:
                eval_path = "preprocessing/ternary_training/{y}/sys/converted/Eval_TTCR_{s}_{y}_0.awkd".format(y=year, s=sample)
                print ("********************* Evaluating {} *****************************".format(eval_path))
                PN_output= (model.predict(eval_dataset.X))
                predicted_class = np.array([np.argmax(PN_output, axis=1)]).T
                nrows, ncolumns = (np.shape(PN_output))
                predicted_probabilites = []
                for row in range(nrows):
                    predicted_probabilites.append(PN_output[row][predicted_class[row]])
                
                ofile_path = inputfilepath[region][year] + systype + '/' + sysdir + '/' + inputfilename[region][sample][:-len('.root')]+'_'+systype+'_'+sysdir+'_test.root'
                rootIO.add_fourbranches(ofile_path, treename, 'jetchargetagger_prob_nodeWp', 'F', np.array(PN_output[:,0]).flatten(), 'jetchargetagger_prob_nodeWn', 'F', np.array(PN_output[:,1]).flatten(),'jetchargetagger_prob_nodeZ', 'F', np.array(PN_output[:,2]).flatten(), 'jetchargetagger_ind', 'F', predicted_class.flatten())
                print ("Branches added to the root file {}".format(ofile_path))

def main():

    global year
    predict_testset()
    #predict_mc()
    #predict_data()
    #predict_sys()   

if __name__ == '__main__':
    main()
