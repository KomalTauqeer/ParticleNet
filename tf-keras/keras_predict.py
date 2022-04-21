import numpy as np
import awkward

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

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

    def __init__(self, filepath, feature_dict = {}, label='label', pad_len=100, data_format='channel_first'):
        self.filepath = filepath
        self.feature_dict = feature_dict
        if len(feature_dict)==0:
            feature_dict['points'] = ['part_etarel', 'part_phirel']
            feature_dict['features'] = ['part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel', 'part_charge', 'part_deltaR']
            feature_dict['mask'] = ['part_pt_log']
        self.label = label
        self.pad_len = pad_len
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        self._values = {}
        self._label = None
        self._load()

    def _load(self):
        logging.info('Start loading file %s' % self.filepath)
        counts = None
        with awkward.load(self.filepath) as a:
            self._label = a[self.label]
            for k in self.feature_dict:
                cols = self.feature_dict[k]
                if not isinstance(cols, (list, tuple)):
                    cols = [cols]
                arrs = []
                for col in cols:
                    if counts is None:
                        counts = a[col].counts
                    else:
                        assert np.array_equal(counts, a[col].counts)
                    arrs.append(pad_array(a[col], self.pad_len))
                self._values[k] = np.stack(arrs, axis=self.stack_axis)
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

test_dataset = Dataset('preprocessing/converted/test_ssWWVBS_0.awkd', data_format='channel_last')
#test_dataset = Dataset('preprocessing/converted/test_file_0.awkd', data_format='channel_last')
#test_dataset = Dataset('tutorial_datasets/converted/test_file_0.awkd', data_format='channel_last')

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import roc_curve,RocCurveDisplay,auc

#Load model
#model = keras.models.load_model("particle_net_lite_lite_checkpoints/particle_net_lite_model.030.h5")
#model = keras.models.load_model("with1addvar_particle_net_lite_lite_checkpoints/particle_net_lite_model.028.h5")
#model = keras.models.load_model("with2addvar_particle_net_lite_lite_checkpoints/particle_net_lite_model.030.h5")
#model = keras.models.load_model("with3addvar_particle_net_lite_lite_checkpoints/particle_net_lite_model.025.h5")
#model = keras.models.load_model("with3addvar_particle_net_lite_checkpoints/particle_net_lite_model.012.h5")
model = keras.models.load_model("particle_net_lite_checkpoints/particle_net_lite_model.019.h5")

test_dataset.shuffle()

#PN_output= (model.predict(test_dataset.X)).round()
PN_output= (model.predict(test_dataset.X))

truth_labels = test_dataset.y

print ("Output: ")
print (PN_output)
print ("Truth: ")
print (truth_labels)


#Plot DNN output
plt.hist(PN_output[truth_labels[:,0]==1,0],30,histtype='step',color='red',label='$\mathrm{W^+}$')
plt.hist(PN_output[truth_labels[:,0]==0,0],30,histtype='step',color='blue',label='$\mathrm{W^-}$')
plt.legend(loc='upper right')
plt.ylabel('Events')
plt.xlabel('Particle Net score')
plt.savefig('PNLite_ssWWVBS_score.pdf')
plt.close()


#Plot Confusion Matrix
normalized_cm = confusion_matrix(truth_labels.argmax(axis=1),PN_output.argmax(axis=1),normalize = 'true')
unnormalized_cm = confusion_matrix(truth_labels.argmax(axis=1),PN_output.argmax(axis=1))
#print (cm)
cm = ConfusionMatrixDisplay(normalized_cm, display_labels=['$\mathrm{W^+}$','$\mathrm{W^-}$'])
cm.plot()
plt.title('Normalized Confusion Matrix')
plt.savefig('PNLite_CM_ssWWVBS_normalized.pdf')
plt.clf()

cm1 = ConfusionMatrixDisplay(unnormalized_cm,display_labels=['$\mathrm{W^+}$','$\mathrm{W^-}$'])
cm1.plot()
plt.title('Confusion Matrix')
plt.savefig('PNLite_CM_ssWWVBS_unnormalized.pdf')
plt.close()


#Plot ROC curve
#fpr, tpr, thresholds = roc_curve(truth_labels.argmax(axis=1), PN_output.argmax(axis=1)) #,pos_label=1)
#roc_auc = auc(fpr,tpr)
#roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name= 'Particle Net Lite')
#roc.plot()
#plt.title('Receiver operating characteristic (ROC)')
#plt.savefig('ROC.pdf')
#plt.close()

n_classes =1

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(truth_labels[:, i], PN_output[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('PNLite_ROC_ssWWVBS_%s.pdf' % i)

