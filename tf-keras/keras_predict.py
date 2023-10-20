import os
import numpy as np
from numpy import savetxt
import awkward
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import roc_curve,RocCurveDisplay,auc
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
from keras_train import stack_arrays, pad_array, Dataset

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

def plot_output_score(output_score, truth_labels, ofile):
    plt.hist(output_score[truth_labels[:,0]==1,0],30,histtype='step',color='red',label='$\mathrm{W^+}$')
    plt.hist(output_score[truth_labels[:,0]==0,0],30,histtype='step',color='blue',label='$\mathrm{W^-}$')
    plt.legend(loc='upper right')
    plt.ylabel('Events')
    plt.xlabel('Jet charge tagger score')
    plt.savefig(ofile+'_outputscore.pdf')
    plt.clf()
    plt.close()

def plot_confusion_matrix(output_score, truth_labels, ofile):
    normalized_cm = confusion_matrix(truth_labels.argmax(axis=1),output_score.argmax(axis=1),normalize = 'true')
    unnormalized_cm = confusion_matrix(truth_labels.argmax(axis=1),output_score.argmax(axis=1))
    cm = ConfusionMatrixDisplay(normalized_cm, display_labels=['$\mathrm{W^+}$','$\mathrm{W^-}$'])
    cm.plot()
    plt.title('Normalized Confusion Matrix')
    plt.savefig(ofile+'_normalizedCM.pdf')
    plt.clf()
    cm1 = ConfusionMatrixDisplay(unnormalized_cm,display_labels=['$\mathrm{W^+}$','$\mathrm{W^-}$'])
    cm1.plot()
    plt.title('Confusion Matrix')
    plt.savefig(ofile+'_unnormalizedCM.pdf')
    plt.clf()
    plt.close()

def compute_ROC_curve(output_score, truth_labels, ofile):
    n_classes =1
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(truth_labels[:, i], output_score[:, i])
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
        plt.savefig(ofile+'_ROC_{}.pdf'.format(i))
    
def main():
    outdir = 'eval_results'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
 
    model = 'PNL_WpWn_models_Oct20/particle_net_lite_model.030.h5'
    file_to_eval = 'preprocessing/converted/Test_TT_UL18_0.awkd'

    predicted_scores, true_scores = eval_test_file(file_to_eval, model)
    plot_output_score(predicted_scores, true_scores, outdir+'/'+'PNL_WpWn_testTT_m30')
    plot_confusion_matrix(predicted_scores, true_scores, outdir+'/'+'PNL_WpWn_testTT_m30')
    compute_ROC_curve(predicted_scores, true_scores, outdir+'/'+'PNL_WpWn_testTT_m30')

if __name__ == "__main__":
    main()
