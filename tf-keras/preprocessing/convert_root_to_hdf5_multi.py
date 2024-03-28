# activate vir env: conda activate tf
import os
import sys
import optparse
#import uproot4
import awkward as ak
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle
#local imports
import meta_data
from data import *

parser = optparse.OptionParser()
parser.add_option("--train" , "--train", action="store_true", dest = "do_train", help = "train mode", default = False)
parser.add_option("--eval" , "--eval", action="store_true", dest = "do_eval", help = "eval mode", default = False)
parser.add_option("--eval_data" , "--eval_data", action="store_true", dest = "do_eval_data", help = "eval data", default = False)
parser.add_option("--eval_sys" , "--eval_sys", action="store_true", dest = "do_eval_sys", help = "eval sys files", default = False)
parser.add_option("--sys_type", "--sys_type", dest="sys_type", default= None)
parser.add_option("--sys_dir", "--sys_dir", dest="sys_dir", default= None)
parser.add_option("--year", "--y", dest="year", default= "UL18")
parser.add_option("--region", "--r", dest="region", default= None)
parser.add_option("--sample", "--s", dest="sample", default= None)
parser.add_option("--outdir", "--odir", dest="outdir", default= "original")
(options,args) = parser.parse_args()
def main():
    year = options.year
    region = options.region
    sample = options.sample
    outdir = options.outdir
    systype = options.sys_type
    sysdir = options.sys_dir

    if options.do_train:
        filepathTT = meta_data.inputfilepath['TTCR'][year]
        filenameTT = meta_data.inputfilename['TTCR']['TT']
        filepathZJets = meta_data.inputfilepath['ZJetsCR'][year]
        filenameZJets = meta_data.inputfilename['ZJetsCR']['ZJets']
    
    if options.do_eval:
        if region is not None and sample is not None:
            filepath = meta_data.inputfilepath[region][year]
            filename = meta_data.inputfilename[region][sample]
        else:
            print ("You must give the region and the sample argument to prepare the dataset for the evalution. For example: --region=VBSSR --sample=ssWW")
            sys.exit()
    
    if options.do_eval_data:
        if region is not None and sample is not None:
            filepath = meta_data.inputfilepath[region][year]
            if year!= "UL18": filename = meta_data.datafilename[region][sample]
            else: filename = meta_data.datafilename_UL18[region][sample]
        else:
            print ("You must give the region and the sample argument to prepare the dataset for the evalution. For example: --region=VBSSR --sample=Data_muon/Data_electron")
            sys.exit()
    
    if options.do_eval_sys:
        filenames = []
        samples = []
        if systype is not None and sysdir is not None:
            filepath = meta_data.inputfilepath['VBSSR'][year]+systype+'/'+sysdir+'/'
            for sample in list(meta_data.inputfilename_sys['VBSSR']):
                samples.append(sample)
                filename = meta_data.inputfilename_sys['VBSSR'][sample][:-len('_dnn_sr1_0p6.root')]+'_'+systype+'_'+sysdir+'_dnn_sr1_0p6.root'
                filenames.append(filename)
        else:
            print ("You must give --systype and --sysdir for reading sys samples")
            sys.exit()
        
    
    treename = meta_data.treename
    inputvariables = meta_data.variables 
    #if options.do_train: labels = meta_data.labels
    labels = meta_data.labels #I need to read labels in all cases as the Class Dataset the loads the awkd file for PN needs labels.

    if options.do_train:
        opath = outdir + '/multitraining_sets'
        if not os.path.isdir(opath):
            os.makedirs(opath)
  
        print ("***Converting {} file to pandas dataframe***".format(filepathTT+filenameTT))
        TT_df = prepare_input_multitrain(filepathTT, filenameTT, treename, 'TT', inputvariables, labels)
        print(TT_df)
        print ("***Converting {} file to pandas dataframe***".format(filepathZJets+filenameZJets))
        ZJets_df = prepare_input_multitrain(filepathZJets, filenameZJets, treename, 'ZJets', inputvariables, labels)
        print(ZJets_df)

        if (len(TT_df.columns) > len(ZJets_df.columns)):
            train_data = merge_df(TT_df, ZJets_df)
        else: train_data = merge_df(ZJets_df, TT_df)

        df_train, df_val, df_test = split_stratified_into_train_val_test(train_data, stratify_colname=labels[0], frac_train=0.60, frac_val=0.20, frac_test=0.20, random_state=42)
        
        #train_data = shuffle(train_data, random_state=42)
        #df_train, df_valandtest = train_test_split(train_data, test_size=0.4)
        #df_val, df_test = train_test_split(df_valandtest, test_size=0.5)

        print (df_train)
        #Look for class imbalance
        print ("Number of each label in training set:")
        print ("Number of W+: {} ".format(df_train[labels[0]].value_counts()[-1.0]))
        print ("Number of W-: {} ".format(df_train[labels[0]].value_counts()[1.0]))
        print ("Number of Z: {} ".format(df_train[labels[0]].value_counts()[0.0]))
        print ("Number of W+: {} ".format(df_val[labels[0]].value_counts()[-1.0]))
        print ("Number of W-: {} ".format(df_val[labels[0]].value_counts()[1.0]))
        print ("Number of Z: {} ".format(df_val[labels[0]].value_counts()[0.0]))
        print ("Number of W+: {} ".format(df_test[labels[0]].value_counts()[-1.0]))
        print ("Number of W-: {} ".format(df_test[labels[0]].value_counts()[1.0]))
        print ("Number of Z: {} ".format(df_test[labels[0]].value_counts()[0.0]))
        save_dataset(df_train, opath, "WpWnZ_train_{}".format(year))
        save_dataset(df_val, opath, "WpWnZ_val_{}".format(year))
        save_dataset(df_test, opath, "WpWnZ_test_{}".format(year))
        print ("***Multi-training input files for \"{}\" are saved in \"{}\" dir in hdf5 format***".format(year,opath))

    if options.do_eval:
        opath = outdir + '/eval_sets'
        if not os.path.isdir(opath):
            os.makedirs(opath)
        print ("***Converting {} file to pandas dataframe***".format(filepath+filename))
        df = prepare_input_eval(filepath, filename, treename, inputvariables, labels)
        #print ("Eval dataset: \n" + df)
        print (df)
        save_dataset(df, opath, "{}_{}_{}_eval".format(region, sample, year))
        print ("***Eval file for {} is saved in {}***".format(sample,opath))

    if options.do_eval_data:
        opath = outdir + '/data_eval_sets'
        if not os.path.isdir(opath):
            os.makedirs(opath)
        print ("***Converting {} file to pandas dataframe***".format(filepath+filename))
        df = prepare_input_eval(filepath, filename, treename, inputvariables, labels)
        #print ("Eval dataset: \n" + df)
        print (df)
        save_dataset(df, opath, "{}_{}_{}_eval".format(region, sample, year))
        print ("***Eval file for {} is saved in {}***".format(sample,opath))

    if options.do_eval_sys:
        opath = outdir + '/sys'
        if not os.path.isdir(opath):
            os.makedirs(opath)
        for filename in filenames:
            print ("***Converting {} file to pandas dataframe***".format(filepath+filename))
            df = prepare_input_eval(filepath, filename, treename, inputvariables, labels)
            print (df)
            save_dataset(df, opath, "{}_{}_{}_{}_{}_eval".format('VBSSR', samples[filenames.index(filename)], year, systype, sysdir))
            print ("***Eval file for {} is saved in {}***".format(samples[filenames.index(filename)]+'_'+systype+'_'+sysdir,opath))

if __name__ == "__main__":
    main()

