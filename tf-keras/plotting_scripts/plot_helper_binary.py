import matplotlib.pyplot as plt
import numpy
from numpy import savetxt
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import roc_curve,RocCurveDisplay,auc

def plot_output_score(output_score, truth_labels, ofile):
    plt.rcParams['xaxis.labellocation'] = 'right'
    plt.hist(output_score[truth_labels[:,0]==1,0],30,histtype='step',linewidth=2,color='red',label='$\mathrm{W^+}$')
    plt.hist(output_score[truth_labels[:,0]==0,0],30,histtype='step',linewidth=2,color='blue',label='$\mathrm{W^-}$')
    legend_handles = [
        Line2D([0], [0], color='red', linewidth=2, label='$\mathrm{W^+}$'),
        Line2D([0], [0], color='blue', linewidth=2, label='$\mathrm{W^-}$')
    ]
    plt.legend(handles=legend_handles, loc='upper right', fontsize='medium')
    plt.ylabel('Events', fontsize='large')
    plt.xlabel('Jet charge tagger score', fontsize='large')
    plt.savefig(ofile+'_outputscore.pdf')
    plt.clf()
    plt.close()

def plot_confusion_matrix(output_score, truth_labels, ofile):
    plt.rcParams['xaxis.labellocation'] = 'center'
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
    plt.rcParams['xaxis.labellocation'] = 'right'
    n_classes = 1

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
        plt.xlabel('1 - background rejection', fontsize ='large')
        plt.ylabel('Signal efficiency', fontsize='large')
        plt.title('Receiver operating characteristic (ROC)')
        plt.legend(loc="lower right" , fontsize = 'medium')
        plt.savefig(ofile+'_ROC_{}.pdf'.format(i))
    #savetxt('TPR_WpWn_UL18.csv', tpr[0], delimiter=',')
    #savetxt('FPR_WpWn_UL18.csv', fpr[0], delimiter=',')

