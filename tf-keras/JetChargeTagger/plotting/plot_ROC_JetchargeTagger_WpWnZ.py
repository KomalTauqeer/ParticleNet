#This script only works in mplhep_env and with source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-centos7-gcc11-opt/setup.sh for ROOT

import os
import sys
import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#plt.rcParams.update({'font.size': 12})
plt.rcParams["mathtext.fontset"] = "cm"
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import roc_curve,RocCurveDisplay,auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from itertools import cycle, combinations
from ROOT import *
from array import array
import mplhep as hep
import optparse

parser = optparse.OptionParser()
parser.add_option("--year", "--y", dest="year", default= "UL17")
(options,args) = parser.parse_args()
year = options.year

# Load style sheet
plt.style.use(hep.style.CMS) 

def plot_roc_OvsO_scheme_seperate(nclasses, predicted_score, true_score, odir):
    true_labels = np.argmax(true_score, axis=1)

    pair_list = list(combinations(np.unique(true_labels), 2))
    print(pair_list)

    pair_scores = []
    mean_tpr = dict()
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    
    for ix, (label_a, label_b) in enumerate(pair_list):
        print (label_a)
        print (label_b)
        
        a_mask = true_labels == label_a
        b_mask = true_labels == label_b
        ab_mask = np.logical_or(a_mask, b_mask)
        print (a_mask)
        a_true = a_mask[ab_mask]
        #print (a_true)
        b_true = b_mask[ab_mask]

        #My labels are the index values so using them directly
        
        fpr_a, tpr_a, _ = roc_curve(a_true, predicted_score[ab_mask, label_a])
        fpr_b, tpr_b, _ = roc_curve(b_true, predicted_score[ab_mask, label_b])
        auc_a = auc(fpr_a, tpr_a)
        auc_b = auc(fpr_b, tpr_b)
    
        mean_tpr[ix] = np.zeros_like(fpr_grid)
        mean_tpr[ix] += np.interp(fpr_grid, fpr_a, tpr_a)
        mean_tpr[ix] += np.interp(fpr_grid, fpr_b, tpr_b)
        mean_tpr[ix] /= 2
        mean_score = auc(fpr_grid, mean_tpr[ix])
        pair_scores.append(mean_score)
 
        labels_string = {0: "$W^{+}$", 1: "$W^{-}$", 2: "$Z$"}
        #Plot separate OvsO curves for each pair 
        fig, ax = plt.subplots(figsize=(6, 6))
        #plt.plot(
        #    mean_tpr[ix],
        #    1-fpr_grid,
        #    label=f"{label_a} vs {label_b} (AUC = {mean_score :.2f})",
        #    linestyle=":",
        #    linewidth=2,
        #)
        plt.plot(tpr_a, 1-fpr_a, lw=2, label="M+B (AUC = {:.2f})".format(auc_a))
        plt.plot([0, 1], [1, 0], 'k--', lw=2, label="No discrimination")
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.ylabel(r'1 - ({} efficiency)'.format(labels_string[label_b]), fontsize = 'large')
        plt.xlabel(r'{} efficiency'.format(labels_string[label_a]), fontsize = 'large')
        plt.legend()
        #plt.plot(tpr_b, 1-fpr_b, lw=2)
        
        plt.savefig(f'{odir}/{label_a}_vs_{label_b}.pdf')       
        plt.close() 
        

def plot_roc_OvsO_scheme(nclasses, predicted_score, true_score, odir):
    true_labels = np.argmax(true_score, axis=1)

    pair_list = list(combinations(np.unique(true_labels), 2))
    print(pair_list)

    pair_scores = []
    mean_tpr = dict()
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    
    for ix, (label_a, label_b) in enumerate(pair_list):
        print (label_a)
        print (label_b)
        
        a_mask = true_labels == label_a
        b_mask = true_labels == label_b
        ab_mask = np.logical_or(a_mask, b_mask)
        print (a_mask)
        a_true = a_mask[ab_mask]
        #print (a_true)
        b_true = b_mask[ab_mask]

        #My labels are the index values so using them directly
        
        fpr_a, tpr_a, _ = roc_curve(a_true, predicted_score[ab_mask, label_a])
        fpr_b, tpr_b, _ = roc_curve(b_true, predicted_score[ab_mask, label_b])
    
        mean_tpr[ix] = np.zeros_like(fpr_grid)
        mean_tpr[ix] += np.interp(fpr_grid, fpr_a, tpr_a)
        mean_tpr[ix] += np.interp(fpr_grid, fpr_b, tpr_b)
        mean_tpr[ix] /= 2
        mean_score = auc(fpr_grid, mean_tpr[ix])
        pair_scores.append(mean_score)
 
    macro_roc_auc_ovo = roc_auc_score(
        true_labels,
        predicted_score,
        multi_class="ovo",
        average="macro",
    )
    print(f"Macro-averaged One-vs-One ROC AUC score:\n{macro_roc_auc_ovo:.2f}")

    labels_string = {0: "$W^{+}$", 1: "$W^{-}$", 2: "$Z$"}
    #Plot all OvsO together 
    ovo_tpr = np.zeros_like(fpr_grid)
    #fig, ax = plt.subplots(figsize=(6, 6))
   
    color = {0:'purple', 1:'teal', 2:'brown'}
    linestyles = {0:'solid', 1:'dotted', 2:'dashed'}
    
    for ix, (label_a, label_b) in enumerate(pair_list):
        ovo_tpr += mean_tpr[ix]
        #ax.plot(
        #    fpr_grid,
        #    mean_tpr[ix],
        #    label=f"{labels_string[label_a]} vs {labels_string[label_b]} (area = {pair_scores[ix]:.2f})",
        #    linewidth=3,
        #)
        plt.plot(
            fpr_grid,
            mean_tpr[ix],
            label=f"{labels_string[label_a]} vs {labels_string[label_b]} (AUC = {pair_scores[ix]:.2f})",
            linewidth=3,
            color=color[ix],
            linestyle=linestyles[ix],
        )
    
    ovo_tpr /= sum(1 for pair in enumerate(pair_list))
    
    #ax.plot(
    #    fpr_grid,
    #    ovo_tpr,
    #    label=f"One-vs-One macro-average (area = {macro_roc_auc_ovo:.2f})",
    #    linestyle=":",
    #    linewidth=3,
    #)
    plt.plot([0, 1], [0, 1], "k--", label="Random", lw=3)
    #_ = ax.set(
    #    xlabel="1 - background rejection",
    #    ylabel="Signal efficiency",
    #    title="",
    #    aspect="equal",
    #    xlim=(0., 1.),
    #    ylim=(0., 1.05),
    #)
    #ax.xaxis.get_label().set_fontsize(12)
    #ax.yaxis.get_label().set_fontsize(12)
    #ax.legend()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(r'1 - background rejection', fontsize = 'medium')
    plt.ylabel(r'Signal efficiency', fontsize = 'medium')
    #plt.title('Receiver operating characteristic for multi-class')
    plt.legend(loc="lower right",fontsize='small')
    hep.cms.label('Preliminary', data=False,loc=2)
    plt.savefig("{}/ROC_OvsO.pdf".format(odir))
    plt.close()

def roc_curves_multi(predicted_score, true_score, n_classes, odir):

    plot_labels = {0: '$W^{+}$ vs $W^{-}, Z$', 1: '$W^{-}$ vs $W^{+}, Z$', 2: '$Z$ vs $W^{+}, W^{-}$'}

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


    lw = 3
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
    #plt.plot(fpr["micro"], tpr["micro"],
    #         label='micro-average ROC curve  (area = {0:0.2f})'
    #               ''.format(roc_auc["micro"]),
    #         color='deeppink', linestyle=':', linewidth=1)

    #plt.plot(fpr["macro"], tpr["macro"],
    #         label='macro-average ROC curve (area = {0:0.2f})'
    #               ''.format(roc_auc["macro"]),
    #         color='navy', linestyle=':', linewidth=1)

    colors = cycle(['red', 'blue', 'green'])
    linestyles = {0: 'dashed', 1: 'dashdot', 2: 'solid'}
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, linestyle=linestyles[i],
                 label= plot_labels[i] +'  (AUC = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw, label= 'Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(r'1 - background rejection', fontsize = 'medium')
    plt.ylabel(r'Signal efficiency', fontsize = 'medium')
    #plt.title('Receiver operating characteristic for multi-class')
    plt.legend(loc="lower right",fontsize='small')
    hep.cms.label('Preliminary', data=False, loc=2)
    #plt.title(r"CMS $\bf{Work In Progress}$", loc='left', fontsize= 'large', weight='bold')
    #plt.show()
    plt.savefig('{}/ROC_multi.pdf'.format(odir))
    plt.close()
    plt.clf()
    
    #for i, color in zip(range(n_classes), colors):
    #    plt.plot(tpr[i], 1-fpr[i], color=color, lw=lw,
    #             label='ROC curve of '+ plot_labels[i] +'  (area = {1:0.2f})'
    #             ''.format(i, roc_auc[i]))

    #plt.plot([0, 1], [1, 0], 'k--', lw=lw)
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel(r'$\epsilon_{s}$', fontsize = 'xx-large')
    #plt.ylabel(r'$1 - \epsilon_{b}$', fontsize = 'xx-large')
    ##plt.title('Receiver operating characteristic for multi-class')
    #plt.legend(loc="lower left",fontsize='small')
    #plt.title(r"$\bf{CMS}$ Work In Progress", loc='left')
    ##plt.show()
    ##plt.savefig('{}/ROC_eff_multi.pdf'.format(odir))
    #plt.close()
    #plt.clf()

def main():

    #Read root file containing the tagger output and trueclass index from the test dataset

    ifile = TFile.Open("../ternary_training/{}/WpWnZ_test.root".format(year), 'READ') #file created by keras_predict_multi.py for test dataset
    itree = ifile.Get('AnalysisTree')
    nentries = int(itree.GetEntries())

    taggerscoresWp = []
    taggerscoresWn = []
    taggerscoresZ = []
    truescores_ind = []
    nodeWp_reader = array('d', [0])
    nodeWn_reader = array('d', [0])
    nodeZ_reader = array('d', [0])
    truelabels_reader = array('d', [0])
    itree.SetBranchAddress("jetchargetagger_prob_nodeWp", nodeWp_reader)
    itree.SetBranchAddress("jetchargetagger_prob_nodeWn", nodeWn_reader)
    itree.SetBranchAddress("jetchargetagger_prob_nodeZ", nodeZ_reader)
    itree.SetBranchAddress("jetchargetagger_true_ind", truelabels_reader)
    for ientry in range(nentries):
        itree.GetEntry(ientry)
        taggerscoresWp.append(nodeWp_reader[0])
        taggerscoresWn.append(nodeWn_reader[0])
        taggerscoresZ.append(nodeZ_reader[0])
        truescores_ind.append(truelabels_reader[0])
    ifile.Close()
    PN_output = np.transpose([np.array(taggerscoresWp), np.array(taggerscoresWn), np.array(taggerscoresZ)])
    truescores_ind = np.array(truescores_ind)
    true_scores = []
    for i in truescores_ind:
        if i == 0.: true_scores.append([1, 0, 0])
        if i == 1.: true_scores.append([0, 1, 0])
        if i == 2.: true_scores.append([0, 0, 1])
    true_scores = np.array(true_scores)
    print (PN_output)
    print (truescores_ind)
    print (true_scores)
    
    outdir= './ROCPlotsMulti/{}'.format(year)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
   
    if roc_multi: 
        roc_curves_multi(PN_output, true_scores, 3, outdir)
        plot_roc_OvsO_scheme(3, PN_output, true_scores, outdir)
        #plot_roc_OvsO_scheme_seperate(3, PN_output, true_labels, outdir)


if __name__ == '__main__':
    roc_multi = True
    main()
