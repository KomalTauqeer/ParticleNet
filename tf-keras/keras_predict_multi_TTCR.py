import os
import sys
sys.path.append("preprocessing/")
import numpy as np
import optparse
import tensorflow as tf
from tensorflow import keras
from load_datasets import Dataset
import rootIO
from meta_data_new import *

parser = optparse.OptionParser()
parser.add_option("--year", dest="year", default= "UL18")
parser.add_option("--region", dest="region", default= 'TTCR')
parser.add_option("--do_sys" , "--do_sys", action="store_true", dest = "do_sys", help = "predict jec and jer files", default = False)
parser.add_option("--do_eval" , "--do_eval", action="store_true", dest = "do_eval", help = "predict on MC + data samples", default = False)
(options,args) = parser.parse_args()
year = options.year
region = options.region

samples = ["TT", "ST", "WJet", "QCD"]
data_samples = ["Data_singlemuon", "Data_singleelectron"]


def main():

    #Load model
    model = keras.models.load_model("ternary_training/130624_manualSplitting/model_checkpoints/particle_net_lite_model.027.h5")
 
    if options.do_eval:
        for sample in data_samples+samples:
            print ("Evalutating file: preprocessing/converted/eval/{}/Eval_{}_{}_{}_0.awkd".format(year, region, sample, year))
            eval_dataset = Dataset('preprocessing/converted/eval/{}/Eval_{}_{}_{}_0.awkd'.format(year, region, sample, year), data_format='channel_last')
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
            if "Data" in sample and year == "UL18": file_path = inputfilepath[region][year] + datafilename_UL18[region][sample].rstrip('.root') + '_test.root'
            elif "Data" in sample and year!= "UL18": file_path = inputfilepath[region][year] + datafilename[region][sample].rstrip('.root') + '_test.root'
            else: file_path = inputfilepath[region][year] + inputfilename[region][sample].rstrip('.root') + '_test.root'

            rootIO.add_branches(file_path, treename, 'jetchargetagger_prob', 'F', np.array(predicted_probabilites).flatten(), 'jetchargetagger_ind', 'F', predicted_class.flatten())
            print ("Branches added to the root file {}".format(file_path))
    
    if options.do_sys:
       for systype in ["jec", "jer"]:
           for sysdir in ["up", "down"]:
               for sample in samples:
                   print ("Evaluating file: preprocessing/converted/sys/Eval_{}_{}_{}_{}_{}_0.awkd".format(region, sample, year, systype, sysdir))
                   eval_dataset = Dataset('preprocessing/converted/sys/Eval_{}_{}_{}_{}_{}_0.awkd'.format(region, sample, year, systype, sysdir), data_format='channel_last')
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
                   file_path = inputfilepath[region][year] + systype + '/' + sysdir + '/' + inputfilename[region][sample][:-len('.root')]+'_'+systype+'_'+sysdir+'_test.root'
                   rootIO.add_branches(file_path, treename, 'jetchargetagger_prob', 'F', np.array(predicted_probabilites).flatten(), 'jetchargetagger_ind', 'F', predicted_class.flatten())
                   print ("Branches added to the root file {}".format(file_path))


if __name__ == '__main__':
    main()
