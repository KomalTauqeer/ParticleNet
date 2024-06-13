#use virtual env: conda activate tf_py36. Do get_root to source ROOT.
# Usage: python keras_test.py --model=past_trainings/training_results_Oct23_lrsch_1e-3/model_checkpoints/particle_net_lite_model.025.h5  --srcdir=preprocessing/converted/test/UL18/ --file=Test_TTCR_Data_singleelectron_UL18

import os
import sys
sys.path.append("plotting_scripts")
sys.path.append("preprocessing")
import argparse
import numpy as np
from numpy import savetxt
import awkward
import tensorflow as tf
from tensorflow import keras
import matplotlib as plt
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
from plot_helper_binary import *
from array import array
import ROOT
from ROOT import *
from load_datasets import *
import meta_data_new as meta_data
import rootIO

parser = argparse.ArgumentParser()
parser.add_argument("--mode", "--n", dest="mode", default=None, help ="test/eval: will test the binary training test file/will test the file corresponding to the region and sample name" )
parser.add_argument("--region", "--r", dest="region", default="TTCR" )
parser.add_argument("--sample", "--s", dest="sample", default= "TT")
parser.add_argument("--year", "--y", dest="year", default="UL18")
parser.add_argument("--save_root_file" , "--save_root_file", action="store_true", dest = "save_root_file", help = "Store the dnnout, labels and weights in a new root file", default = False)
parser.add_argument("--do_sys" , "--do_sys", action="store_true", dest = "do_sys", help = "convert jec and jer files", default = False)
parser.add_argument("--sys_type", "--sys_type", dest="sys_type", default= None)
parser.add_argument("--sys_dir", "--sys_dir", dest="sys_dir", default= None)

args = parser.parse_args()
mode = args.mode
region = args.region
sample = args.sample
year = args.year
save_root_file = args.save_root_file
systype = args.sys_type
sysdir = args.sys_dir

if mode == 'eval' and (region == 'None' or sample == 'None'): 
    print ("You must give the region and sample name to eval the file using --region and --sample flags.")
    sys.exit()

def eval_test_file(testfile, model_path):
    test_dataset = Dataset(testfile, data_format='channel_last')
    model = keras.models.load_model(model_path)
    test_dataset.shuffle()
    model_output= model.predict(test_dataset.X)
    truth_labels = test_dataset.y
    weights = test_dataset.Weights
    #print (model_output)
    #print (truth_labels)
    #print (weights)
    return model_output, truth_labels, weights

def main():
    outdir = 'test_results_binary' if mode == 'test' else 'eval_results_binary'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
 
    #modelpath = 'past_trainings/training_results_Oct23_lrsch_1e-3/model_checkpoints/particle_net_lite_model.025.h5'
    #modelpath = 'binary_training/280424/model_checkpoints/particle_net_lite_model.026.h5'
    modelpath = 'binary_training/290424_customsplitting/model_checkpoints/particle_net_lite_model.024.h5'
   
    if args.do_sys:
        file_to_test = 'preprocessing/converted/test/{}/Test_{}_{}_{}_{}_{}_0.awkd'.format(year,region,sample,year,systype,sysdir)

    else: 
        file_to_test = 'preprocessing/converted/binary_training/{}/WpWn_test_UL18_0.awkd'.format(year) if mode == 'test' else 'preprocessing/converted/test/{}/Test_{}_{}_{}_0.awkd'.format(year,region,sample, year)

    predicted_scores, true_scores, evt_weights = eval_test_file(file_to_test, modelpath)

    print(predicted_scores[:,0])
    print(len(predicted_scores[:,0]))
    print(true_scores[:,0])
    print(evt_weights)
    
    #Make some plots to validate
    if mode == 'eval':
        plot_output_score(predicted_scores, true_scores, outdir+'/{region}_{sample}_{year}')
        plot_confusion_matrix(predicted_scores, true_scores, outdir+'/{region}_{sample}_{year}')
        compute_ROC_curve(predicted_scores, true_scores, outdir+'/{region}_{sample}_{year}')
    elif mode == 'test':
        plot_output_score(predicted_scores, true_scores, outdir+'/testfile_{year}')
        plot_confusion_matrix(predicted_scores, true_scores, outdir+'/testfile_{year}')
        compute_ROC_curve(predicted_scores, true_scores, outdir+'/testfile_{year}')

    if args.do_sys and save_root_file:
        filepath = meta_data.inputfilepath[region][year]+systype + '/' +sysdir + '/'
        filename = meta_data.inputfilename[region][sample][:-len('.root')]+'_'+systype+'_'+sysdir+'_test.root'
        print (filepath+filename)
        rootIO.add_branches(filepath+filename, "AnalysisTree", "jetchargetagger_WpWn", "F", predicted_scores[:,0], "true_scores_WpWn", "F", true_scores[:,0])

    if save_root_file and mode == 'test':
        filepathTT = meta_data.inputfilepath['TTCR'][year]
        filenameTT = meta_data.inputfilename['TTCR']['TT'].rstrip('.root') + '_test.root'
        rootIO.add_branches(filepathTT+filenameTT, "AnalysisTree", "jetchargetagger_WpWn", "F", predicted_scores[:,0], "true_scores_WpWn", "F", true_scores[:,0])

    if save_root_file and mode == 'eval':
        if region is not None and sample is not None:
            filepath = meta_data.inputfilepath[region][year]
            if "Data_" in sample:
                if year!= "UL18": filename = meta_data.datafilename[region][sample].rstrip('.root') + '_test.root'
                else: filename = meta_data.datafilename_UL18[region][sample].rstrip('.root') + '_test.root'
            else: filename = meta_data.inputfilename[region][sample]
            rootIO.add_branches(filepath+filename, "AnalysisTree", "jetchargetagger_WpWn", "F", predicted_scores[:,0], "true_scores_WpWn", "F", true_scores[:,0])
        else:
  
            print ("You must give the region and the sample argument to prepare the dataset for the evalution. For example: --region=VBSSR --sample=ssWW")
        sys.exit()


if __name__ == "__main__":
    main()
