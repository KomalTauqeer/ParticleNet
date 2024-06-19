# activate vir env: conda activate tf_py36
#Purpose: convert already split root files using split_files_manullay.py to hdf5 format

import os
import sys
import optparse

#local imports
import meta_data_new as meta_data
import data_utils
from data_utils import *

parser = optparse.OptionParser()
parser.add_option("--mode" , "--mode", dest = "mode", help = "binary/ternary training", default = None)
parser.add_option("--train" , "--train", action="store_true", dest = "do_train", help = "train mode", default = False)
parser.add_option("--eval" , "--eval", action="store_true", dest = "do_eval", help = "eval mode, can be applied on samples with no labels. This flag is independent of mode", default = False)
parser.add_option("--test" , "--test", action="store_true", dest = "do_test", help = "test mode, labelled data is required. This flag is only available in binary classification mode. Currently, used to test VBS samples with binary classification", default = False)
parser.add_option("--year", "--y", dest="year", help = "UL16preVFP, UL16postVFP, UL17, UL18", default= "UL18")
parser.add_option("--region", "--r", dest="region", help = "TTCR, VBSSR, ZJetsCR", default= None)
parser.add_option("--sample", "--s", dest="sample", help = "Name of the sample, see meta_data.py", default= None)
parser.add_option("--outdir", "--odir", dest="outdir", help = "Specify the output directory", default= "original")
parser.add_option("--do_sys" , "--do_sys", action="store_true", dest = "do_sys", help = "convert jec and jer files", default = False)
parser.add_option("--sys_type", "--sys_type", dest="sys_type", default= None)
parser.add_option("--sys_dir", "--sys_dir", dest="sys_dir", default= None)
parser.add_option("--eval_sys" , "--eval_sys", action="store_true", dest = "do_eval_sys", help = "eval sys files", default = False)

(options,args) = parser.parse_args()
year = options.year
region = options.region
sample = options.sample
outdir = options.outdir
mode = options.mode
systype = options.sys_type
sysdir = options.sys_dir

if options.do_train and mode == "binary":
    filepathTT = meta_data.inputfilepath['TTCR'][year]
    filenameTT = meta_data.inputfilename['TTCR']['TT']
elif options.do_train and mode == "ternary":
    filepathTT = meta_data.inputfilepath['TTCR'][year]
    filenameTT = meta_data.inputfilename['TTCR']['TT']
    filepathZJets = meta_data.inputfilepath['ZJetsCR'][year]
    filenameZJets = meta_data.inputfilename['ZJetsCR']['ZJets']

elif options.do_test and mode == "binary": #Only converts test dataset for testing
    if region == 'TTCR':
        filepath = meta_data.inputfilepath[region][year]
        if year!= "UL18": filename = meta_data.datafilename[region][sample].rstrip('.root') + '_test.root'
        else: filename = meta_data.datafilename_UL18[region][sample].rstrip('.root') + '_test.root'
    else:
        print ("You must give the region and the sample argument to prepare the dataset for the evalution. For example: --region=VBSSR --sample=ssWW")
        sys.exit()

elif options.do_sys and mode == "binary": #Only converts test part
    if systype is not None and sysdir is not None:
        filepath = meta_data.inputfilepath[region][year]+systype+'/'+sysdir+'/'
        filename = meta_data.inputfilename[region][sample][:-len('.root')]+'_'+systype+'_'+sysdir+'_test.root'
    else: sys.exit("Please give sys_type and sys_dir!")

elif options.do_eval: #Only converts test part
    if region is not None and sample is not None:
        filepath = meta_data.inputfilepath[region][year]
        if "Data_" in sample:
            if year!= "UL18": filename = meta_data.datafilename[region][sample].rstrip('.root') + '_test.root'
            else: filename = meta_data.datafilename_UL18[region][sample].rstrip('.root') + '_test.root'
        else: filename = meta_data.inputfilename[region][sample].rstrip('.root') + '_test.root'
    else:
        print ("You must give the region and the sample argument to prepare the dataset for the evalution. For example: --region=VBSSR --sample=ssWW")
        sys.exit()

elif options.do_eval_sys: #Only converts test part
    if region is not None and sample is not None and systype is not None and sysdir is not None:
        filepath = meta_data.inputfilepath[region][year]+systype+'/'+sysdir+'/'
        filename = meta_data.inputfilename[region][sample][:-len('.root')]+'_'+systype+'_'+sysdir+'_test.root'
    else:
        print ("You must give --region, --sample, --systype, and --sysdir for reading sys samples")
        sys.exit()


treename = meta_data.treename
inputvariables = meta_data.variables
labels = meta_data.labels
weights = meta_data.weights

if mode == "binary" and options.do_train:

    train_filename = filenameTT.rstrip('.root') + '_train.root'
    val_filename = filenameTT.rstrip('.root') + '_val.root'
    test_filename = filenameTT.rstrip('.root') + '_test.root'

    train_file = filepathTT + train_filename
    val_file = filepathTT + val_filename
    test_file = filepathTT + test_filename

    print ("***Converting {} file to pandas dataframe***".format(train_file))
    df_train = prepare_input_dataset_binarytrain(filepathTT, train_filename, treename, inputvariables, labels, weights)
    print (df_train)
    print ("Total number of W+: {} ".format(df_train[labels[0]].value_counts()[-1.0]))
    print ("Total number of W-: {} ".format(df_train[labels[0]].value_counts()[1.0]))
    
    print ("***Converting {} file to pandas dataframe***".format(val_file))
    df_val = prepare_input_dataset_binarytrain(filepathTT, val_filename, treename, inputvariables, labels, weights)
    print (df_val)
    print ("Total number of W+: {} ".format(df_val[labels[0]].value_counts()[-1.0]))
    print ("Total number of W-: {} ".format(df_val[labels[0]].value_counts()[1.0]))
    
    print ("***Converting {} file to pandas dataframe***".format(test_file))
    df_test = prepare_input_dataset_binarytrain(filepathTT, test_filename, treename, inputvariables, labels, weights)
    print (df_test)
    print ("Total number of W+: {} ".format(df_test[labels[0]].value_counts()[-1.0]))
    print ("Total number of W-: {} ".format(df_test[labels[0]].value_counts()[1.0]))

    opath = outdir + '/binary_training/{}'.format(year) + '/' 
    if not os.path.isdir(opath):
        os.makedirs(opath)
    save_dataset(df_train, opath, 'WpWn_train_{}'.format(year))
    save_dataset(df_val, opath, 'WpWn_val_{}'.format(year))
    save_dataset(df_test, opath, 'WpWn_test_{}'.format(year))
    print ("***Binary-training input files for \"{}\" are saved in \"{}\" dir in hdf5 format***".format(year,opath))

elif mode == "ternary" and options.do_train:
    train_filenameTT = filenameTT.rstrip('.root') + '_train.root'
    val_filenameTT = filenameTT.rstrip('.root') + '_val.root'
    test_filenameTT = filenameTT.rstrip('.root') + '_test.root'

    train_fileTT = filepathTT + train_filenameTT
    val_fileTT = filepathTT + val_filenameTT
    test_fileTT = filepathTT + test_filenameTT

    train_filenameZJets = filenameZJets.rstrip('.root') + '_train.root'
    val_filenameZJets = filenameZJets.rstrip('.root') + '_val.root'
    test_filenameZJets = filenameZJets.rstrip('.root') + '_test.root'

    train_fileZJets = filepathZJets + train_filenameZJets
    val_fileZJets = filepathZJets + val_filenameZJets
    test_fileZJets = filepathZJets + test_filenameZJets

    print ("***Converting {} file to pandas dataframe***".format(train_fileTT))
    TT_df_train = prepare_input_dataset_multitrain(filepathTT, train_filenameTT, treename, 'TT', inputvariables, labels, weights)
    print ("***Converting {} file to pandas dataframe***".format(val_fileTT))
    TT_df_val = prepare_input_dataset_multitrain(filepathTT, val_filenameTT, treename, 'TT', inputvariables, labels, weights)
    print ("***Converting {} file to pandas dataframe***".format(test_fileTT))
    TT_df_test = prepare_input_dataset_multitrain(filepathTT, test_filenameTT, treename, 'TT', inputvariables, labels, weights)
    
    print ("***Converting {} file to pandas dataframe***".format(train_fileZJets))
    ZJets_df_train = prepare_input_dataset_multitrain(filepathZJets, train_filenameZJets, treename, 'ZJets', inputvariables, labels, weights)
    print ("***Converting {} file to pandas dataframe***".format(val_fileZJets))
    ZJets_df_val = prepare_input_dataset_multitrain(filepathZJets, val_filenameZJets, treename, 'ZJets', inputvariables, labels, weights)
    print ("***Converting {} file to pandas dataframe***".format(test_fileZJets))
    ZJets_df_test = prepare_input_dataset_multitrain(filepathZJets, test_filenameZJets, treename, 'ZJets', inputvariables, labels, weights)

    if (len(TT_df_train.columns) > len(ZJets_df_train.columns)):
        train_data = merge_df(TT_df_train, ZJets_df_train)
    else: train_data = merge_df(ZJets_df_train, TT_df_train)

    if (len(TT_df_val.columns) > len(ZJets_df_val.columns)):
        val_data = merge_df(TT_df_val, ZJets_df_val)
    else: val_data = merge_df(ZJets_df_val, TT_df_val)

    if (len(TT_df_test.columns) > len(ZJets_df_test.columns)):
        test_data = merge_df(TT_df_test, ZJets_df_test)
    else: test_data = merge_df(ZJets_df_test, TT_df_test)
   
    train_data = shuffle(train_data, random_state=42)
    val_data = shuffle(val_data, random_state=42)
    test_data = shuffle(test_data, random_state=42)
    
    print (train_data)
    print (val_data)
    print (test_data)

    print ("Number of W+ in train: {} ".format(train_data[labels[0]].value_counts()[-1.0]))
    print ("Number of W- in train: {} ".format(train_data[labels[0]].value_counts()[1.0]))
    print ("Number of Z in train: {} ".format(train_data[labels[0]].value_counts()[0.0]))
    print ("Number of W+ in val: {} ".format(val_data[labels[0]].value_counts()[-1.0]))
    print ("Number of W- in val: {} ".format(val_data[labels[0]].value_counts()[1.0]))
    print ("Number of Z in val: {} ".format(val_data[labels[0]].value_counts()[0.0]))
    print ("Number of W+ in test: {} ".format(test_data[labels[0]].value_counts()[-1.0]))
    print ("Number of W- in test: {} ".format(test_data[labels[0]].value_counts()[1.0]))
    print ("Number of Z in test: {} ".format(test_data[labels[0]].value_counts()[0.0]))

    opath = outdir + '/ternary_training/{}'.format(year) + '/' 
    if not os.path.isdir(opath):
        os.makedirs(opath)
    save_dataset(train_data, opath, 'WpWnZ_train_{}'.format(year))
    save_dataset(val_data, opath, 'WpWnZ_val_{}'.format(year))
    save_dataset(test_data, opath, 'WpWnZ_test_{}'.format(year))
    print ("***Ternary-training input files for \"{}\" are saved in \"{}\" dir in hdf5 format***".format(year,opath))

elif mode == "binary" and options.do_test:
    print ("***Converting {} file to pandas dataframe***".format(filepath+filename))
    test_dataset = prepare_input_test(filepath, filename, treename, inputvariables, labels, weights) 
    #test_dataset = shuffle(test_dataset, random_state=42)
    opath = outdir + '/test/{}'.format(year) + '/'
    if not os.path.isdir(opath):
        os.makedirs(opath)
    save_dataset(test_dataset, opath, "Test_{}_{}_{}".format(region, sample, year))
    print ("***Test files for \"{}\" are saved in \"{}\" dir in hdf5 format***".format(filename,opath))

elif mode == "binary" and options.do_sys:
    print ("***Converting {} file to pandas dataframe***".format(filepath+filename))
    test_dataset = prepare_input_test(filepath, filename, treename, inputvariables, labels, weights)
    opath = outdir + '/test/{}'.format(year) + '/'
    if not os.path.isdir(opath):
        os.makedirs(opath)
    save_dataset(test_dataset, opath, "Test_{}_{}_{}_{}_{}".format(region, sample, year, systype ,sysdir))
    print ("***Sys files for \"{}\" are saved in \"{}\" dir in hdf5 format***".format(filename,opath))

elif options.do_eval: 
    print ("***Converting {} file to pandas dataframe***".format(filepath+filename))
    eval_dataset = prepare_input_eval(filepath, filename, treename, inputvariables, labels, weights) 
    eval_dataset = shuffle(eval_dataset, random_state=42)
    opath = outdir + '/eval/{}'.format(year) + '/'
    if not os.path.isdir(opath):
        os.makedirs(opath)
    save_dataset(eval_dataset, opath, "Eval_{}_{}_{}".format(region, sample, year))
    print ("***Eval files for \"{}\" are saved in \"{}\" dir in hdf5 format***".format(filename,opath))

elif options.do_eval_sys:
    opath = outdir + '/sys'
    if not os.path.isdir(opath):
        os.makedirs(opath)
    print ("***Converting {} file to pandas dataframe***".format(filepath+filename))
    df = prepare_input_eval(filepath, filename, treename, inputvariables, labels, weights)
    print (df)
    save_dataset(df, opath, "{}_{}_{}_{}_{}_eval".format(region, sample, year, systype, sysdir))
    print ("***Eval file for {} is saved in {}***".format(sample+'_'+systype+'_'+sysdir,opath))


else: 
    print ("No valid option to process!")

