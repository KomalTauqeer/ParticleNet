# activate vir env: conda activate tf
import os
import sys
import optparse

#local imports
import meta_data
import data_utils
from data_utils import *

parser = optparse.OptionParser()
parser.add_option("--mode" , "--mode", dest = "mode", help = "binary/ternary", default = "binary")
parser.add_option("--train" , "--train", action="store_true", dest = "do_train", help = "train mode", default = False)
parser.add_option("--eval" , "--eval", action="store_true", dest = "do_eval", help = "eval mode, can be applied on samples with no labels", default = False)
parser.add_option("--test" , "--test", action="store_true", dest = "do_test", help = "test mode, labelled data is required", default = False)
parser.add_option("--year", "--y", dest="year", help = "UL16preVFP, UL16postVFP, UL17, UL18", default= "UL18")
parser.add_option("--region", "--r", dest="region", help = "TTCR, VBSSR, ZJetsCR", default= None)
parser.add_option("--sample", "--s", dest="sample", help = "Name of the sample, see meta_data.py", default= None)
parser.add_option("--outdir", "--odir", dest="outdir", help = "Specify the output directory", default= "original")

(options,args) = parser.parse_args()
year = options.year
region = options.region
sample = options.sample
outdir = options.outdir
mode = options.mode

if options.do_train and mode == "binary":
    filepathTT = meta_data.inputfilepath['TTCR'][year]
    filenameTT = meta_data.inputfilename['TTCR']['TT']
elif options.do_train and mode == "ternary":
    filepathTT = meta_data.inputfilepath['TTCR'][year]
    filenameTT = meta_data.inputfilename['TTCR']['TT']
    filepathZJets = meta_data.inputfilepath['ZJetsCR'][year]
    filenameZJets = meta_data.inputfilename['ZJetsCR']['ZJets']

if options.do_eval or options.do_test:
    if region is not None and sample is not None:
        filepath = meta_data.inputfilepath[region][year]
        if "Data_" in sample:
            if year!= "UL18": filename = meta_data.datafilename[region][sample]
            else: filename = meta_data.datafilename_UL18[region][sample]
        else: filename = meta_data.inputfilename[region][sample]
    else:
        print ("You must give the region and the sample argument to prepare the dataset for the evalution. For example: --region=VBSSR --sample=ssWW")
        sys.exit()

treename = meta_data.treename
inputvariables = meta_data.variables
labels = meta_data.labels
weights = meta_data.weights

if mode == "binary" and options.do_train:
    print ("***Converting {} file to pandas dataframe***".format(filepathTT+filenameTT))
    train_data = prepare_input_dataset_binarytrain(filepathTT, filenameTT, treename, inputvariables, labels, weights)
    train_data = shuffle(train_data, random_state=42)
    print (train_data)
    print ("Total number of W+: {} ".format(train_data[labels[0]].value_counts()[-1.0]))
    print ("Total number of W-: {} ".format(train_data[labels[0]].value_counts()[1.0]))
    df_train, df_valandtest = train_test_split(train_data, test_size=0.4)
    df_val, df_test = train_test_split(df_valandtest, test_size=0.5)
    opath = outdir + '/binary_training/{}'.format(year) + '/' 
    if not os.path.isdir(opath):
        os.makedirs(opath)
    save_dataset(df_train, opath, 'WpWn_train_{}'.format(year))
    save_dataset(df_val, opath, 'WpWn_val_{}'.format(year))
    save_dataset(df_test, opath, 'WpWn_test_{}'.format(year))
    print ("***Binary-training input files for \"{}\" are saved in \"{}\" dir in hdf5 format***".format(year,opath))

elif mode == "ternary" and options.do_train:
    print ("***Converting {} file to pandas dataframe***".format(filepathTT+filenameTT))
    TT_df = prepare_input_dataset_multitrain(filepathTT, filenameTT, treename, 'TT', inputvariables, labels, weights)
    print (TT_df)
    print ("***Converting {} file to pandas dataframe***".format(filepathZJets+filenameZJets))
    ZJets_df = prepare_input_dataset_multitrain(filepathZJets, filenameZJets, treename, 'ZJets', inputvariables, labels, weights)
    print (ZJets_df)
    if (len(TT_df.columns) > len(ZJets_df.columns)):
        train_data = merge_df(TT_df, ZJets_df)
    else: train_data = merge_df(ZJets_df, TT_df)
    #In TT process sign of W boson is -1* sign of lepton
    print ("Number of W+: {} ".format(train_data[labels[0]].value_counts()[-1.0]))
    print ("Number of W-: {} ".format(train_data[labels[0]].value_counts()[1.0]))
    print ("Number of Z: {} ".format(train_data[labels[0]].value_counts()[0.0]))
    df_train, df_val, df_test = split_stratified_into_train_val_test(train_data, stratify_colname=labels[0], frac_train=0.60, frac_val=0.20, frac_test=0.20, random_state=42)
    opath = outdir + '/ternary_training/{}'.format(year) + '/'
    if not os.path.isdir(opath):
        os.makedirs(opath)
    save_dataset(df_train, opath, "WpWnZ_train_{}".format(year))
    save_dataset(df_val, opath, "WpWnZ_val_{}".format(year))
    save_dataset(df_test, opath, "WpWnZ_test_{}".format(year))
    print ("***Multi-training input files for \"{}\" are saved in \"{}\" dir in hdf5 format***".format(year,opath))

elif options.do_test:
    print ("***Converting {} file to pandas dataframe***".format(filepath+filename))
    test_dataset = prepare_input_test(filepath, filename, treename, inputvariables, labels, weights) 
    test_dataset = shuffle(test_dataset, random_state=42)
    opath = outdir + '/test/{}'.format(year) + '/'
    if not os.path.isdir(opath):
        os.makedirs(opath)
    save_dataset(test_dataset, opath, "Test_{}_{}_{}".format(region, sample, year))
    print ("***Test files for \"{}\" are saved in \"{}\" dir in hdf5 format***".format(filename,opath))

elif options.do_eval:
    print ("***Converting {} file to pandas dataframe***".format(filepath+filename))
    eval_dataset = prepare_input_eval(filepath, filename, treename, inputvariables, weights) 
    eval_dataset = shuffle(eval_dataset, random_state=42)
    opath = outdir + '/eval/{}'.format(year) + '/'
    if not os.path.isdir(opath):
        os.makedirs(opath)
    save_dataset(eval_dataset, opath, "Eval_{}_{}_{}".format(region, sample, year))
    print ("***Eval files for \"{}\" are saved in \"{}\" dir in hdf5 format***".format(filename,opath))

else: 
    print ("No valid option to process!")

