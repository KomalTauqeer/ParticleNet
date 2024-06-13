import os
import sys
import datetime
import optparse
import numpy as np
import awkward
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from load_datasets import *
import shap

parser = optparse.OptionParser()
parser.add_option("--year", "--y", dest="year", default= "UL18")
parser.add_option("--outdir", "--outdir", dest="outdir", default= "shap_values")
(options,args) = parser.parse_args()
year= options.year
outdir = options.outdir

def get_shap_values():
    #Load training and test dataset
    train_dataset = Dataset('preprocessing/converted/btag_test/multitraining_sets/WpWnZ_genmatched_train_{}_0.awkd'.format(year), data_format='channel_last')
    test_dataset = Dataset('preprocessing/converted/btag_test/multitraining_sets/WpWnZ_genmatched_test_{}_0.awkd'.format(year), data_format='channel_last')
    model = keras.models.load_model("culrsch2_test_training/model_checkpoints/particle_net_lite_model.020.h5")

    train_dataset.shuffle()
    test_dataset.shuffle()
 
    #Convert dict to np array
    print (np.shape(train_dataset.X['points']))
    print (np.shape(train_dataset.X['features']))
    print (np.shape(train_dataset.X['add_features']))
   
    data1 = train_dataset.X['points']
    data2 = train_dataset.X['features']
    data3 = train_dataset.X['add_features']
    background_data = np.array([data1[100::], data2[100::], data3[100:]])
    print (background_data) 
    print (np.shape(background_data))

    #e = shap.DeepExplainer(model, train_dataset.X)
    #shap_values = e.shap_values(test_dataset.X) 
    #print(shap_values)      


def main():
    get_shap_values()

if __name__ == "__main__":
    main()

