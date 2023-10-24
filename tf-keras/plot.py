# imports from standard python
from __future__ import print_function
import os
import sys
import time
from multiprocessing import Process
import matplotlib
matplotlib.use('PDF')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.style.use('./CMSStyle.py')
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np
import scipy
#from tqdm import tqdm


def clear(iplot):
    iplot.clf()
    iplot.cla()
    iplot.close()


def inputvars(df_s, df_b, opath, vlist):

    print('--- Creating plots for all input variables...')

    #plotpath = './' + opath + '/plots'
    plotpath = './plots/' + opath 
    if not os.path.isdir(plotpath):
        os.mkdir(plotpath)

    plot_processes = []

    for v in vlist:
        plot_processes.append(Process(target=inputvars_process,
                                      args=(df_s[v], df_b[v], plotpath, v)))

    for p in plot_processes:
        # artificial delay to keep the os happy
        time.sleep(0.05)
        p.start()

    for p in plot_processes:
        p.join()


def inputvars_process(sig, bkg, plotpath, var):
    bins=np.linspace(min(sig), max(bkg), 30)

    fig = plt.figure(figsize=(5,5))
    p = fig.add_subplot(111)

    p.hist(sig, density=True, bins=bins, alpha=0.6, histtype='stepfilled', label='signal', color='C1')
    p.hist(bkg, density=True, bins=bins, alpha=0.6, histtype='stepfilled', label='background', color='C0')

    p.set_title('input variable')
    p.legend(loc=0)
    p.set_xlabel(var.replace('$','\$'), horizontalalignment='right', x=1.0)
    p.set_ylabel('probability density', horizontalalignment='right', y=1.0)

    fig.tight_layout()
    fig.savefig(plotpath + '/plot_' + var + '.pdf')
    plt.close()

def inputvars_multitrain(df_1, df_2, df_3, vlist):

    print('--- Creating plots for all input variables...')

    plotpath = './inputplots_multi/'  
    if not os.path.isdir(plotpath):
        os.mkdir(plotpath)

    plot_processes = []

    for v in vlist:
        plot_processes.append(Process(target=inputvars_multitrain_process,
                                      args=(df_1[v], df_2[v], df_3[v], plotpath, v)))

    for p in plot_processes:
        # artificial delay to keep the os happy
        time.sleep(0.05)
        p.start()

    for p in plot_processes:
        p.join()

def inputvars_multitrain_process(df_1, df_2, df_3, plotpath, var):
    bins=np.linspace(-2, 2, 30)

    fig = plt.figure(figsize=(5,5))
    p = fig.add_subplot(111)

    p.hist(df_1, density=True, bins=bins, alpha=0.6, histtype='step', label='W^{+}', color='C3')
    p.hist(df_2, density=True, bins=bins, alpha=0.6, histtype='step', label='W^{-}', color='C0')
    p.hist(df_3, density=True, bins=bins, alpha=0.6, histtype='step', label='Z', color='C2')

    p.set_title('Jet charge tagger input variable')
    p.legend(loc=0)
    p.set_xlabel(var.replace('$','\$'), horizontalalignment='right', x=1.0)
    p.set_ylabel('probability density', horizontalalignment='right', y=1.0)

    fig.tight_layout()
    fig.savefig(plotpath + '/plot_' + var + '.pdf')
    plt.close()

def roc(fpr, tpr, auroc, opath, name):
    fig = plt.figure(figsize=(5,5))
    p = fig.add_subplot(111)

    p.plot(fpr, tpr, label='ROC curve (area = {:.3f})'.format(auroc))
    p.plot([0, 1], [0, 1], 'k-', alpha=0.7, linewidth=0.5)

    p.set_title('receiver operating characteristic')
    p.legend(loc="lower right")
    p.set_xlabel('false positive rate', horizontalalignment='right', x=1.0)
    p.set_ylabel('true positive rate', horizontalalignment='right', y=1.0)

    p.set_xlim([0., 1.])
    p.set_ylim([0., 1.])

    fig.tight_layout()
    fig.savefig(opath + '/roc_' + name + '.pdf')
    plt.close()

def correlation(df, opath, vlist, sample):
    #df = df.drop(df.index[0],axis=1)
    df.columns.name = None
    print('--- Creating plots for correlation matrix of input variables...')
    matplotlib.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots()
    corr = df.corr()
    im = ax.imshow(corr, interpolation="nearest", origin="lower", cmap=plt.get_cmap("bwr"))
    ax.set_xticks(np.arange(len(vlist)))
    ax.set_yticks(np.arange(len(vlist)))
    ax.set_xticklabels(vlist, fontsize=8)
    ax.set_yticklabels(vlist, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(vlist)):
        for j in range(len(vlist)):
            text = ax.text(j, i, round(corr.iat[i,j],2), ha="center", va="center", color="black", fontsize=5)
    if sample == "sig":
        ax.set_title("Correlation matrix: signal")
    else:
        ax.set_title("Correlation matrix: background")
    fig.colorbar(im,pad=0.01)
    #fig.tight_layout()
    im.set_clim(-1.0,1.0)
    plt.subplots_adjust(left=0.3, bottom=0.25, right=0.99, top=0.95)
    if sample == "sig":
        plt.savefig('plots/'+opath+'/correlation_sig.pdf')
    else:
        plt.savefig('plots/'+opath + '/correlation_bkg.pdf')
    clear(plt)

def training_history(train_dict, opath):
    h = pd.DataFrame()
    h['epoch'] = range(len(train_dict['loss']))
    h['loss'] = train_dict['loss']
    h['val_loss'] = train_dict['val_loss']
    h['accuracy'] = train_dict['accuracy']
    h['val_accuracy'] = train_dict['val_accuracy']

    h.to_csv(opath + '/history.csv')

    fig = plt.figure(figsize=(5,5))
    p = fig.add_subplot(111)

    p.plot(h['epoch'], h['val_loss'], '-', label='Validation loss')
    p.plot(h['epoch'], h['loss'], '-', label='Training loss')


    p.set_title('model loss')
    p.legend(loc=0)
    p.set_xlabel('epoch', horizontalalignment='right', x=1.0)
    p.set_ylabel('loss', horizontalalignment='right', y=1.0)

    fig.tight_layout()
    fig.savefig(opath + '/model_loss.pdf')
    plt.close()



    fig = plt.figure(figsize=(5,5))
    p = fig.add_subplot(111)

    p.plot(h['epoch'], h['val_accuracy'], '-', label='Validation accuracy')
    p.plot(h['epoch'], h['accuracy'], '-', label='Training accuracy')


    p.set_title('model accuracy')
    p.legend(loc=0)
    p.set_xlabel('epoch', horizontalalignment='right', x=1.0)
    p.set_ylabel('accuracyuracy', horizontalalignment='right', y=1.0)

    p.set_ylim(top=1.01)

    fig.tight_layout()
    fig.savefig(opath + '/model_accuracy.pdf')
    plt.close()

