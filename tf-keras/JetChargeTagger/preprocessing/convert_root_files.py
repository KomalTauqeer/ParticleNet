# activate vir env: conda activate tf_py36
#Purpose: convert already split root files using split_files_manullay.py to hdf5 format, then convert them into .awkd format while also calculating tagger input variables

import os
import sys
import optparse
#local imports
from constants import *
from data_utils import *
from prepare_tagger_inputs import convert

parser = optparse.OptionParser()
parser.add_option("--year", "--y", dest="year", help = "UL16preVFP, UL16postVFP, UL17, UL18", default= "UL17")
(options,args) = parser.parse_args()

year = options.year
valid_years = ["UL16preVFP", "UL16postVFP", "UL17", "UL18"]
if year not in valid_years:
    raise ValueError(f"Invalid year: {year}. Must be one of {', '.join(valid_years)}.")

def get_TTMC(year):
    fileTT = inputfilepath['TTCR'][year] + inputfilename['TTCR']['TT']
    TT_train = fileTT[:-len('.root')]+'_train.root'
    TT_val = fileTT[:-len('.root')]+'_val.root'
    TT_test = fileTT[:-len('.root')]+'_test.root'
    
    print ("***Converting {} file to pandas dataframe***".format(TT_train))
    df_train = prepare_input_dataset(TT_train, treename, "TT", inputvariables, labels, eventWeights["TT"])
    print ("***Converting {} file to pandas dataframe***".format(TT_val))
    df_val = prepare_input_dataset(TT_val, treename, "TT", inputvariables, labels, eventWeights["TT"])
    print ("***Converting {} file to pandas dataframe***".format(TT_test))
    df_test = prepare_input_dataset(TT_test, treename, "TT", inputvariables, labels, eventWeights["TT"])

    return df_train, df_val, df_test

def get_ZJetsMC(year):
    fileZJets = inputfilepath['ZJetsCR'][year] + inputfilename['ZJetsCR']['ZJets']
    ZJets_train = fileZJets[:-len('.root')]+'_train.root'
    ZJets_val = fileZJets[:-len('.root')]+'_val.root'
    ZJets_test = fileZJets[:-len('.root')]+'_test.root'

    print ("***Converting {} file to pandas dataframe***".format(ZJets_train))
    df_train = prepare_input_dataset(ZJets_train, treename, "ZJets", inputvariables, labels, eventWeights["ZJets"])
    print ("***Converting {} file to pandas dataframe***".format(ZJets_val))
    df_val = prepare_input_dataset(ZJets_val, treename, "ZJets", inputvariables, labels, eventWeights["ZJets"])
    print ("***Converting {} file to pandas dataframe***".format(ZJets_test))
    df_test = prepare_input_dataset(ZJets_test, treename, "ZJets", inputvariables, labels, eventWeights["ZJets"])

    return df_train, df_val, df_test

def prepare_binaryset(year, outdir):

    TT_train, TT_val, TT_test = get_TTMC(year)

    train_data = shuffle(TT_train, random_state=42)
    val_data = shuffle(TT_val, random_state=42)
    test_data = TT_test

    print ("Total number of W+ for training: {} ".format(train_data[labels[0]].value_counts()[-1.0]))
    print ("Total number of W- for training: {} ".format(train_data[labels[0]].value_counts()[1.0]))
    
    opath = outdir+'/binary_training/{}'.format(year) + '/' 
    if not os.path.isdir(opath):
        os.makedirs(opath)
    save_dataset(train_data, opath, 'WpWn_train_{}'.format(year))
    save_dataset(val_data, opath, 'WpWn_val_{}'.format(year))
    save_dataset(test_data, opath, 'WpWn_test_{}'.format(year))
    print ("***Binary-training input files for \"{}\" are saved in \"{}\" dir in hdf5 format***".format(year,opath))
    print ("**************************** Converting hdf5 to awkd *******************************************")
    convert(os.path.join(opath, 'WpWn_train_{}.h5'.format(year)), destdir=opath+'/converted', basename='WpWn_train_{}'.format(year), mode="binary")
    convert(os.path.join(opath, 'WpWn_val_{}.h5'.format(year)), destdir=opath+'/converted', basename='WpWn_val_{}'.format(year), mode="binary")
    convert(os.path.join(opath, 'WpWn_test_{}.h5'.format(year)), destdir=opath+'/converted', basename='WpWn_test_{}'.format(year), mode="binary")
    print ("***Binary-training input files for \"{}\" are saved in \"{}\" dir in awkd format***".format(year,opath+'/converted'))

def prepare_multiset(year, outdir):

    TT_train, TT_val, TT_test  = get_TTMC(year)
    ZJets_train, ZJets_val, ZJets_test  = get_ZJetsMC(year)

    if (len(TT_train.columns) > len(ZJets_train.columns)):
        train_data = merge_df(TT_train, ZJets_train)
    else: train_data = merge_df(ZJets_train, TT_train)

    if (len(TT_val.columns) > len(ZJets_val.columns)):
        val_data = merge_df(TT_val, ZJets_val)
    else: val_data = merge_df(ZJets_val, TT_val)

    if (len(TT_test.columns) > len(ZJets_test.columns)):
        test_data = merge_df(TT_test, ZJets_test)
    else: test_data = merge_df(ZJets_test, TT_test)
   
    train_data = shuffle(train_data, random_state=42)
    val_data = shuffle(val_data, random_state=42)
    
    print (train_data)
    print (val_data)
    print (test_data) #not shuffled

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
    print ("******************************* Converting hd5 to awkd *************************************")
    convert(os.path.join(opath, 'WpWnZ_train_{}.h5'.format(year)), destdir=opath+'/converted', basename='WpWnZ_train_{}'.format(year), mode="multi")
    convert(os.path.join(opath, 'WpWnZ_val_{}.h5'.format(year)), destdir=opath+'/converted', basename='WpWnZ_val_{}'.format(year), mode="multi")
    convert(os.path.join(opath, 'WpWnZ_test_{}.h5'.format(year)), destdir=opath+'/converted', basename='WpWnZ_test_{}'.format(year), mode="multi")
    print ("***Multi-training input files for \"{}\" are saved in \"{}\" dir in awkd format***".format(year,opath+'/converted'))

def prepare_data(year, outdir):
    opath1 = outdir + 'binary_training/{}/Data'.format(year) + '/'
    opath2 = outdir + 'ternary_training/{}/Data'.format(year) + '/'
    if not os.path.isdir(opath1):
        os.makedirs(opath1)
    if not os.path.isdir(opath2):
        os.makedirs(opath2)

    for sample in data_samples:
        filepath = inputfilepath["TTCR"][year]
        if year!= "UL18": filename = datafilename["TTCR"][sample].rstrip('.root') + '_test.root'
        else: filename = datafilename_UL18["TTCR"][sample].rstrip('.root') + '_test.root'
        print ("***Converting {} file to pandas dataframe***".format(filepath+filename))
        df = prepare_input_dataset(filepath+filename, treename, sample, inputvariables, labels, eventWeights[sample])
        
        save_dataset(df, opath1, "Data_TTCR_{}_{}".format(sample, year))
        save_dataset(df, opath2, "Data_TTCR_{}_{}".format(sample, year))
        print ("***Data test files for \"{}\" are saved in \"{}\" dir in hdf5 format***".format(filename,opath1+" and " +opath2))
        convert(os.path.join(opath1, 'Data_TTCR_{}_{}.h5'.format(sample,year)), destdir=opath1+'/converted', basename='Data_TTCR_{}_{}'.format(sample,year), mode="binary")
        convert(os.path.join(opath2, 'Data_TTCR_{}_{}.h5'.format(sample,year)), destdir=opath2+'/converted', basename='Data_TTCR_{}_{}'.format(sample,year), mode="ternary")
        print ("***Data test files for \"{}\" are saved in \"{}\" dir in awkd format***".format(filename,opath1+'/converted and '+opath2+'/converted'))

def prepare_sys(year, outdir):
    opath1 = outdir + 'binary_training/{}/sys'.format(year) + '/'
    opath2 = outdir + 'ternary_training/{}/sys'.format(year) + '/'
    if not os.path.isdir(opath1):
        os.makedirs(opath1)
    if not os.path.isdir(opath2):
        os.makedirs(opath2)

    for sample in samples:
        for unc in ["jec", "jer"]:
            for uncdir in ["up", "down"]:
                filepath = inputfilepath["TTCR"][year] + '/' + unc + '/' + uncdir + '/'
                filename = inputfilename["TTCR"][sample].rstrip('.root') + '_{}_{}_test.root'.format(unc, uncdir)
                print ("***Converting {} file to pandas dataframe***".format(filepath+filename))
                df = prepare_input_dataset(filepath+filename, treename, sample, inputvariables, labels, eventWeights[sample])
                
                save_dataset(df, opath1, "Sys_TTCR_{}_{}_{}_{}".format(sample, year, unc, uncdir))
                save_dataset(df, opath2, "Sys_TTCR_{}_{}_{}_{}".format(sample, year, unc, uncdir))
                print ("***Sys test files for \"{}\" are saved in \"{}\" dir in hdf5 format***".format(filename,opath1 + ' and ' + opath2))
                convert(os.path.join(opath1, 'Sys_TTCR_{}_{}_{}_{}.h5'.format(sample,year, unc, uncdir)), destdir=opath1+'/converted', basename='Sys_TTCR_{}_{}_{}_{}'.format(sample,year, unc, uncdir), mode="binary")
                convert(os.path.join(opath2, 'Sys_TTCR_{}_{}_{}_{}.h5'.format(sample,year, unc, uncdir)), destdir=opath2+'/converted', basename='Sys_TTCR_{}_{}_{}_{}'.format(sample,year, unc, uncdir), mode="ternary")
                print ("***Sys test files for \"{}\" are saved in \"{}\" dir in awkd format***".format(filename,opath1+'/converted and '+opath2+'/converted'))

def prepare_other_testsets(year, outdir):
    opath1 = outdir + 'binary_training/{}/Eval'.format(year) + '/'
    opath2 = outdir + 'ternary_training/{}/Eval'.format(year) + '/'
    if not os.path.isdir(opath1):
        os.makedirs(opath1)
    if not os.path.isdir(opath2):
        os.makedirs(opath2)

    for sample in samples:
        filepath = inputfilepath["TTCR"][year]  
        filename = inputfilename["TTCR"][sample].rstrip('.root') + '_test.root'
        print ("***Converting {} file to pandas dataframe***".format(filepath+filename))
        df = prepare_input_dataset(filepath+filename, treename, sample, inputvariables, labels, eventWeights[sample])
        
        save_dataset(df, opath1, "Eval_TTCR_{}_{}".format(sample, year))
        save_dataset(df, opath2, "Eval_TTCR_{}_{}".format(sample, year))
        print ("***Eval test files for \"{}\" are saved in \"{}\" dir in hdf5 format***".format(filename,opath1 + ' and ' + opath2))
        convert(os.path.join(opath1, 'Eval_TTCR_{}_{}.h5'.format(sample,year)), destdir=opath1+'/converted', basename='Eval_TTCR_{}_{}'.format(sample,year), mode="binary")
        convert(os.path.join(opath2, 'Eval_TTCR_{}_{}.h5'.format(sample,year)), destdir=opath2+'/converted', basename='Eval_TTCR_{}_{}'.format(sample,year), mode="ternary")
        print ("***Eval test files for \"{}\" are saved in \"{}\" dir in awkd format***".format(filename,opath1+'/converted and '+opath2+'/converted'))

def main():
    outdir = './'
    #prepare_binaryset(year, outdir)
    #prepare_multiset(year, outdir)
    #prepare_data(year, outdir)
    #prepare_sys(year, outdir)
    prepare_other_testsets(year, outdir)



if __name__ == "__main__":
    main()
