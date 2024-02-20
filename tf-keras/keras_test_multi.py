import os
import sys
sys.path.append("plotting_scripts/")
import optparse
import numpy as np
from numpy import savetxt
import awkward
import utilis_multiclass
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

parser = optparse.OptionParser()
parser.add_option("--test" , action="store_true", dest = "do_test", help = "test mode", default = True)
parser.add_option("--eval" , action="store_true", dest = "do_eval", help = "eval mode", default = False)
parser.add_option("--year", dest="year", default= "UL18")
parser.add_option("--region", dest="region", default= 'TTCR')
parser.add_option("--sample", dest="sample", default= 'TT')
parser.add_option("--outdir", dest="outdir", default= "evaluation_results")
parser.add_option("--modelpath", dest="modelpath", default= None)
(options,args) = parser.parse_args()
year = options.year
region = options.region
sample = options.sample
outdir = options.outdir
modelpath = options.modelpath


def plot_tagger_output_multi(predicted_score, true_score, node, odir):
    plt.rcParams['xaxis.labellocation'] = 'right'
    node_labels= {0: '$\mathrm{W^+}$', 1: '$\mathrm{W^-}$', 2: '$\mathrm{Z}$'}
    plt.hist(predicted_score[:][np.argmax(true_score, axis=1)==0][:,node],30,histtype='step',color='red',label='$W^+$', linewidth=2)
    plt.hist(predicted_score[:][np.argmax(true_score, axis=1)==1][:,node],30,histtype='step',color='blue',label='$W^-$', linewidth=2)
    plt.hist(predicted_score[:][np.argmax(true_score, axis=1)==2][:,node],30,histtype='step',color='green',label='$Z$', linewidth=2)
    legend_handles = [
        Line2D([0], [0], color='red', linewidth=2, label='$\mathrm{W^+}$'),
        Line2D([0], [0], color='blue', linewidth=2, label='$\mathrm{W^-}$'),
        Line2D([0], [0], color='green', linewidth=2, label='$\mathrm{Z}$'),
    ]

    plt.legend(handles=legend_handles, loc='upper right', fontsize = 'large')
    plt.ylabel('Events', fontsize='large')
    plt.xlabel('Jet charge tagger output [{} node]'.format(node_labels[node]), fontsize='large')
    plt.savefig('{}/PNLite_WpWnZ_score_{}.pdf'.format(odir,node+1))
    plt.close()

#def plot_roc_each_class_seperate(nclasses, predicted_score, true_score, odir):
#    n_classes = nclasses
#    
#    # Compute ROC curve and ROC area for each class
#    fpr = dict()
#    tpr = dict()
#    roc_auc = dict()
#    for i in range(n_classes):
#        fpr[i], tpr[i], _ = roc_curve(true_score[:, i], predicted_score[:, i])
#        roc_auc[i] = auc(fpr[i], tpr[i])
#    
#    # Plot of a ROC curve for a specific class
#    for i in range(n_classes):
#        plt.figure()
#        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
#        plt.plot([0, 1], [0, 1], 'k--')
#        plt.xlim([0.0, 1.0])
#        plt.ylim([0.0, 1.05])
#        plt.xlabel('False Positive Rate')
#        plt.ylabel('True Positive Rate')
#        plt.title('Receiver operating characteristic (ROC)')
#        plt.legend(loc="lower right", fontsize ='small')
#        plt.savefig('{}/ROC_%s.pdf'.format(odir, i))
#    plt.close()

def roc_curves_multi(predicted_score,true_score,n_classes, odir):

    plot_labels = {0: '$W^{+}$', 1: '$W^{-}$', 2: '$Z$'}

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    for i in range(n_classes):
       fpr[i], tpr[i], threshold[i] = roc_curve(true_score[:, i], predicted_score[:, i])
       roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_score.ravel(), predicted_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    lw = 2
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve  (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=1)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=1)

    colors = cycle(['red', 'blue', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of '+ plot_labels[i] +'  (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(r'1 - background rejection', fontsize = 'large')
    plt.ylabel(r'Signal efficiency', fontsize = 'large')
    #plt.title('Receiver operating characteristic for multi-class')
    plt.legend(loc="lower right",fontsize='small')
    #plt.show()
    plt.savefig('{}/ROC_multi.pdf'.format(odir))
    plt.close()
    plt.clf()
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(tpr[i], 1-fpr[i], color=color, lw=lw,
                 label='ROC curve of '+ plot_labels[i] +'  (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [1, 0], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(r'$\epsilon_{s}$', fontsize = 'xx-large')
    plt.ylabel(r'$1 - \epsilon_{b}$', fontsize = 'xx-large')
    #plt.title('Receiver operating characteristic for multi-class')
    plt.legend(loc="lower left",fontsize='small')
    #plt.show()
    plt.savefig('{}/ROC_eff_multi.pdf'.format(odir))
    plt.close()
    plt.clf()

def compute_nsignal_nbackground(predicted_score, true_score, node):
    Wp_score = np.array(predicted_score[:][np.argmax(true_score, axis=1)==0][:,node])
    Wn_score = np.array(predicted_score[:][np.argmax(true_score, axis=1)==1][:,node])
    Z_score = np.array(predicted_score[:][np.argmax(true_score, axis=1)==2][:,node])
    thresholds = []
    signal = []
    background = []
    fpr = []
    tpr = []
    for i in np.arange(0.1,1.,0.01):
        if node == 0:
            n_signal = len(Wp_score[Wp_score<i])
            n_background = len(Wn_score[Wn_score<i]) + len(Z_score[Z_score<i])
        if node == 1:
            n_signal = len(Wn_score[Wn_score<i])
            n_background = len(Wp_score[Wp_score<i]) + len(Z_score[Z_score<i])
        if node == 2:
            n_signal = len(Z_score[Z_score<i])
            n_background = len(Wp_score[Wp_score<i]) + len(Wn_score[Wn_score<i])
        thresholds.append(i)
        signal.append(n_signal)
        background.append(n_background)
        y_prob = predicted_score[:,node]
        y_true = true_score[:,node]
        y_pred = np.where(y_prob >= i, 1, 0)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
    
    return signal, background, thresholds, fpr, tpr

def plot_signal_significance_vs_threshold(thresholds, nsignal, nbackground, node, odir):
    plot_labels = {0: '$W^{+}$', 1: '$W^{-}$', 2: '$Z$'}
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    nsignal = np.array(nsignal)
    nbackground = np.array(nbackground)
    sratiosb = nsignal/np.sqrt(nsignal+nbackground)
    plt.plot(thresholds, sratiosb, lw=2, color=colors[node], label=plot_labels[node])
    plt.xlabel('jet charge tagger threshold value')
    plt.ylabel(r'$S/\sqrt{S+B}$')
    plt.legend(loc="lower right",fontsize='small')
    if node == 2: 
        plt.savefig('{}/signal_significance_vs_threshold.pdf'.format(odir))
        plt.close()
    return sratiosb

def plot_signal_significance_vs_bkgeff(fpr, sratiosb, odir):
    plot_labels = {0: '$W^{+}$', 1: '$W^{-}$', 2: '$Z$'}
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    for node in [0,1,2]:
        plt.plot(np.array(fpr[node]), np.array(sratiosb[node]), lw=2, color=colors[node], label=plot_labels[node])
    
    plt.xlabel('1 - background rejection', fontsize= 'large')
    plt.ylabel(r'$S/\sqrt{S+B}$', fontsize = 'large')
    #plt.legend(loc="upper right",fontsize='small')
    plt.savefig('{}/signal_significance_vs_bkgefficiency.pdf'.format(odir))
    plt.close()

def main():

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if options.do_test:
        #eval_dataset = Dataset('preprocessing/converted/multitraining_sets/WpWnZ_genmatched_test_{}_0.awkd'.format(year), data_format='channel_last')
        eval_dataset = Dataset('preprocessing/converted/ternary_training/{y}/WpWnZ_test_nobtag_{y}_0.awkd'.format(y=year), data_format='channel_last')
    if options.do_eval and options.sample is not None and options.region is not None:
        eval_dataset = Dataset('preprocessing/converted/eval_sets/{}_{}_{}_0.awkd'.format(region, sample, year), data_format='channel_last')
    elif options.do_eval and (options.sample is None or options.region is None):
        print ("Please give sample and region to evaluate the dataset")

    #Load model
    #model = keras.models.load_model("successful_multitrain_results/Nov7_lrsch_1e-4_genmatchedTT/model_checkpoints/particle_net_lite_model.030.h5")
    #retrained genmatchedZ model
    #model = keras.models.load_model("training_history/model_checkpoints/particle_net_lite_model.030.h5")
    #retrained genmatched+btagVars model
    #model = keras.models.load_model("multitrain_addfeatures/model_checkpoints/particle_net_lite_model.029.h5")
    model = keras.models.load_model(modelpath)
    #model used everywhere without gen matching
    #model = keras.models.load_model("successful_multitrain_results/Oct23_lrsch_1e-4/PNL_checkpoints_lrsch_1e-4/particle_net_lite_model.029.h5")
    ##NOT using this: model = keras.models.load_model("successful_multitrain_results/Oct17_lrsch_1e-3/PNL_checkpoints_lrsch_1e-3/particle_net_lite_lite_model.030.h5") 
    
    eval_dataset.shuffle()
    
    PN_output= (model.predict(eval_dataset.X))
    #print ("Output: ")
    print (PN_output)
    predicted_value = np.argmax(PN_output, axis=1)
    
    if options.do_test: 
        true_labels = eval_dataset.y
        #print ("Truth: ")
        #print (true_labels)
        true_value = np.argmax(true_labels, axis=1)
   
        if roc_multi: 
            roc_curves_multi(PN_output, true_labels, 3, outdir)

        if svsb_curves:
            svssb = dict()
            fpr = dict()
            tpr = dict()
            for node in [0,1,2]: 
                signal, background, thresholds, fpr[node], tpr[node] = compute_nsignal_nbackground(PN_output, true_labels, node)
                svssb[node] = plot_signal_significance_vs_threshold(thresholds, signal, background, node, outdir)
            plot_signal_significance_vs_bkgeff(fpr, svssb, outdir)

        if confusion_matrix: utilis_multiclass.plot_confusion_matrix(true_value, predicted_value, outdir, sample_type= 'test', classes=["W+", "W-", "Z"], normalize=True, title='Normalized confusion matrix')

        if report: print (classification_report(true_value, predicted_value, target_names=["W+", "W-", "Z"]))

        if output_score: 
            for node in [0,1,2]: 
                plot_tagger_output_multi(PN_output, true_labels, node, outdir)


if __name__ == '__main__':
    roc_multi = True
    output_score = True
    report = True
    confusion_matrix = True
    svsb_curves = True
    main()
