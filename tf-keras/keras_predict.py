#To run this script we need ROOT built for the python 3.6.8 which is available in test_env enviroment. Make sure to source: "source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.24.08/x86_64-centos7-gcc48-opt/bin/thisroot.sh" 

import os
import sys
import argparse
import numpy as np
from numpy import savetxt
import awkward
import tensorflow as tf
from tensorflow import keras
import matplotlib as plt
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
from array import array
import ROOT
from ROOT import *
from load_datasets import *
import rootIO

parser = argparse.ArgumentParser()
parser.add_argument("--region", "--r", dest="region", default="TTCR" )
parser.add_argument("--sample", "--s", dest="sample", default= "TT")
parser.add_argument("--year", "--y", dest="year", default="UL18")
#parser.add_argument("--model", "--model", dest="model", default=None)
args = parser.parse_args()
region = args.region
sample = args.sample
year = args.year
#modelpath = args.model

def eval(eval_file, model_path):
    eval_dataset = Dataset(eval_file, data_format='channel_last', load_evalset=True)
    model = keras.models.load_model(model_path)
    model_output= model.predict(eval_dataset.X)
    weights = eval_dataset.Weights
    print (model_output)
    return model_output, weights

def main():
    outdir = 'eval_results_lrsch_1e-3'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
 
    model = 'past_trainings/training_results_Oct23_lrsch_1e-3/model_checkpoints/particle_net_lite_model.025.h5'
    
    #file_to_eval = 'preprocessing/converted/eval/{}/Eval_{}_{}_{}_0.awkd'.format(year,region,sample, year)
    file_to_eval = 'preprocessing/converted/test/{}/Test_{}_{}_{}_0.awkd'.format(year,region,sample, year)

    predicted_scores, evt_weights = eval(file_to_eval, model)
    print(predicted_scores[:,0])
    print(len(predicted_scores[:,0]))
    print(evt_weights)

if __name__ == "__main__":
    main()
