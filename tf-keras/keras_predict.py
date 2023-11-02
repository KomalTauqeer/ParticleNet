import os
import argparse
import numpy as np
from numpy import savetxt
import awkward
import tensorflow as tf
from tensorflow import keras
import matplotlib as plt
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
from keras_train import stack_arrays, pad_array, Dataset
from plot import *

parser = argparse.ArgumentParser()
parser.add_argument("--region", "--r", dest="region", default="TTCR" )
parser.add_argument("--sample", "--s", dest="sample", default= "TT")
parser.add_argument("--year", "--y", dest="year", default="UL18")
args = parser.parse_args()
region = args.region
sample = args.sample
year = args.year

def eval_test_file(testfile, model_path):
    test_dataset = Dataset(testfile, data_format='channel_last')
    model = keras.models.load_model(model_path)
    #test_dataset.shuffle()
    model_output= model.predict(test_dataset.X)
    truth_labels = test_dataset.y
    print (model_output)
    print (truth_labels)
    return model_output, truth_labels

def eval(eval_file, model_path):
    eval_dataset = Dataset(eval_file, data_format='channel_last')
    model = keras.models.load_model(model_path)
    #eval_dataset.shuffle()
    model_output= model.predict(eval_dataset.X)
    print (model_output)
    return model_output

def main():
    outdir = 'eval_results_lrsch_1e-3'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
 
    #model = 'PNL_WpWn_models_Oct20/particle_net_lite_model.030.h5'
    #model = 'training_results_Oct23_lrsch_1e-4/model_checkpoints/particle_net_lite_model.030.h5'
    model = 'training_results_Oct23_lrsch_1e-3/model_checkpoints/particle_net_lite_model.025.h5'
    #file_to_eval = 'preprocessing/converted/Test_TT_UL18_0.awkd'
    file_to_eval = 'preprocessing/converted/Test_{}_{}_{}_0.awkd'.format(region, sample, year)

    predicted_scores, true_scores = eval_test_file(file_to_eval, model)
    plot_output_score(predicted_scores, true_scores, outdir+'/'+'PNL_WpWn_test{}_m25_{}'.format(sample, year))
    plot_confusion_matrix(predicted_scores, true_scores, outdir+'/'+'PNL_WpWn_test{}_m25_{}'.format(sample, year))
    compute_ROC_curve(predicted_scores, true_scores, outdir+'/'+'PNL_WpWn_test{}_m25_{}'.format(sample, year))

if __name__ == "__main__":
    main()
