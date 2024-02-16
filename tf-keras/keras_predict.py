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
#from keras_train import stack_arrays, pad_array, Dataset
from plot import *
from array import array
import ROOT
from ROOT import *

def stack_arrays(a, keys, axis=-1):
    flat_arr = np.stack([a[k].flatten() for k in keys], axis=axis)
    return awkward.JaggedArray.fromcounts(a[keys[0]].counts, flat_arr)

def pad_array(a, maxlen, value=0., dtype='float32'):
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = s[:maxlen].astype(dtype)
        x[idx, :len(trunc)] = trunc
    return x

class Dataset(object):

    def __init__(self, filepath, feature_dict = {}, label='label', weight='event_weight', pad_len=100, data_format='channel_first'):
        self.filepath = filepath
        self.feature_dict = feature_dict
        if len(feature_dict)==0:
            feature_dict['points'] = ['part_etarel', 'part_phirel']
            feature_dict['features'] = ['part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel', 'part_charge', 'part_deltaR']
            feature_dict['mask'] = ['part_pt_log']
        self.label = label
        self.weight = weight
        self.pad_len = pad_len
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        self._values = {}
        self._label = None
        self._weight = None
        self._load()

    def _load(self):
        logging.info('Start loading file %s' % self.filepath)
        counts = None
        with awkward.load(self.filepath) as a:
            self._label = a[self.label]
            self._weight = a[self.weight]
            for k in self.feature_dict:
                arrs = []
                if not k == 'add_features':
                    cols = self.feature_dict[k]
                    if not isinstance(cols, (list, tuple)):
                        cols = [cols]
                    for col in cols:
                        if counts is None:
                            counts = a[col].counts
                        else:
                            assert np.array_equal(counts, a[col].counts)
                        arrs.append(pad_array(a[col], self.pad_len))
                else:
                    column = self.feature_dict[k]
                    for col in column:
                        arrs.append(a[col])
                self._values[k] = np.stack(arrs, axis=self.stack_axis)
           #print(self._values['features'])

        logging.info('Finished loading file %s' % self.filepath)

    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        if key==self.label:
            return self._label
        else:
            return self._values[key]

    @property
    def X(self):
        return self._values

    @property
    def Weights(self):
        return self._weight

    @property
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]

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
    weights = test_dataset.Weights
    print (model_output)
    print (truth_labels)
    print (weights)
    return model_output, truth_labels, weights

def eval(eval_file, model_path):
    eval_dataset = Dataset(eval_file, data_format='channel_last')
    model = keras.models.load_model(model_path)
    eval_dataset.shuffle()
    model_output= model.predict(eval_dataset.X)
    print (model_output)
    return model_output

def main():
    outdir = 'eval_results_lrsch_1e-3_wgts'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
 
    #model = 'PNL_WpWn_models_Oct20/particle_net_lite_model.030.h5'
    #model = 'training_results_Oct23_lrsch_1e-4/model_checkpoints/particle_net_lite_model.030.h5'
    model = 'training_results_Oct23_lrsch_1e-3/model_checkpoints/particle_net_lite_model.025.h5'
    #file_to_eval = 'preprocessing/converted/Test_TT_UL18_0.awkd'
    #file_to_eval = 'preprocessing/converted/Test_{}_{}_{}_0.awkd'.format(region, sample, year)
    file_to_eval = 'preprocessing/converted/eventwgts/Test_{}_{}_{}_0.awkd'.format(region,sample, year)

    predicted_scores, true_scores, evt_weights = eval_test_file(file_to_eval, model)
    print(predicted_scores[:,0])
    print(len(predicted_scores[:,0]))
    print(true_scores[:,0])
    print(evt_weights)
    plot_output_score(predicted_scores, true_scores, outdir+'/'+'PNL_WpWn_test{}_m25_{}'.format(sample, year))
    plot_confusion_matrix(predicted_scores, true_scores, outdir+'/'+'PNL_WpWn_test{}_m25_{}'.format(sample, year))
    compute_ROC_curve(predicted_scores, true_scores, outdir+'/'+'PNL_WpWn_test{}_m25_{}'.format(sample, year))

    file1 = TFile.Open('{}_{}_{}_wgts.root'.format(region, sample, year), "RECREATE")
    tree = TTree("AnalysisTree", "AnalysisTree")
    dnnscores = array('d', [0])
    truelabels = array('d', [0])
    weights = array('d', [0])
    tree.Branch('dnnscores', dnnscores, 'dnnscores/D')
    tree.Branch('truelabels', truelabels, 'truelabels/D')
    tree.Branch('event_weight', weights, 'event_weights/D')
    for itr in range(len(predicted_scores[:,0])):
        dnnscores[0] = predicted_scores[itr,0]
        truelabels[0] = true_scores[itr,0]
        weights[0] = evt_weights[itr]
        tree.Fill()
    tree.Write()
    file1.Write()
    file1.Close()

if __name__ == "__main__":
    main()
