# activate vir env: conda activate tf

import os
import sys
sys.path.append('../') 
import uproot4
import awkward as ak
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import plot

def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                         frac_train=0.6, frac_val=0.15, frac_test=0.25,
                                         random_state=None):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.
    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

def root2df(sfile, tree_name, variables, **kwargs):
    f = uproot4.open(sfile)
    tree = f[tree_name]
    df = tree.arrays(variables, library="pd")
    return df

def unstack_multi_df(df):
    df = df.unstack()
    #Remove subentries level and rename dataframe columns 
    df.columns = [a[0] + "_" +str(a[1]) for a in df.columns.to_flat_index()]
    return df

def pandas_to_hdf5(df, outdir, outfilename):
    df.to_hdf(outdir + '/'+ outfilename +'.h5', key='table', mode='w')
    
def save_dataset(dataset, outdir, outfilename):
    pandas_to_hdf5(dataset, outdir, outfilename)

def prepare_input_dataset(filepath, filename, sample_type):
    if sample_type == 'TT' or sample_type == 'VBS':
        variables = ["PF_Px", "PF_Py", "PF_Pz", "PF_E", "PF_q"]
        plotvars = ["PF_q", "PF_logpt", "PF_deta", "PF_dphi", "PF_logE", "PF_logrelE" , "PF_logrelpt", "PF_deltaR", "lep_charge"]
        labels = ["lep_charge"]
        treename = "AnalysisTree"
    elif sample_type == 'ZJets':
        variables = ["PF_Px", "PF_Py", "PF_Pz", "PF_E", "PF_q"]
        plotvars = ["PF_q", "PF_logpt", "PF_deta", "PF_dphi", "PF_logE", "PF_logrelE" , "PF_logrelpt", "PF_deltaR"]
        labels = ["lep_charge"]
        treename = "AnalysisTree"
    elif sample_type == 'JetClassWToQQ':
        variables = ["part_px", "part_py", "part_pz", "part_energy", "part_charge"]
        labels = ["aux_genpart_pid"]
        treename = "tree"
    elif sample_type == 'JetClassZToQQ':
        variables = ["part_px", "part_py", "part_pz", "part_energy", "part_charge"]
        labels = ["aux_genpart_pid"]
        treename = "tree"
    dataset = root2df(filepath+filename, treename, variables)
    plot_dataset = root2df(filepath+filename, treename, plotvars)
    if sample_type == 'TT' or sample_type == 'VBS':
        labels = root2df(filepath+filename, treename, labels)
    dataset = unstack_multi_df(dataset)
    if sample_type == 'ZJets':
        dataset['lep_charge'] = np.array(0.)
        plot_dataset['lep_charge'] = np.array(0.) 
    if sample_type == 'TT' or sample_type == 'VBS': 
        dataset = dataset.join(labels)
    return dataset, plot_dataset

def merge_df(first, second, index_ignore=True):
    df = pd.concat((first, second), ignore_index=index_ignore)
    return df

def shuffle_df(dataset, random_state=42):
    return shuffle(dataset, random_state)

def main():
    TT_df, plot_TT_df = prepare_input_dataset('/ceph/ktauqeer/ULNtuples/UL18/TTCR/', 'TTCR_TTToSemiLeptonic.root', 'TT')
    ZJets_df, plot_ZJets_df = prepare_input_dataset('/ceph/ktauqeer/ULNtuples/UL18/ZJetsCR/', 'ZJetsCR_ZJets_UL18_genmatchedZ_0To645000.root', 'ZJets')
    
    #Prepare dataframes to plot input variables of PN
    Wp_df = plot_TT_df[plot_TT_df['lep_charge']==1.0]
    Wn_df = plot_TT_df[plot_TT_df['lep_charge']==-1.0]
    Z_df = plot_ZJets_df
    varlist = ["PF_q", "PF_logpt", "PF_deta", "PF_dphi", "PF_logE", "PF_logrelE" , "PF_logrelpt", "PF_deltaR"]
    plot.inputvars_multitrain(Wp_df, Wn_df, Z_df, varlist)
    print ("Input variables plotted......")

    #Merge two dataframe and split in train, test and val sets
    data = merge_df(TT_df, ZJets_df)
    df_train, df_val, df_test = split_stratified_into_train_val_test(data, stratify_colname='lep_charge', frac_train=0.60, frac_val=0.20, frac_test=0.20, random_state=42)
    print (df_train)
    pandas_to_hdf5(df_train, "original", "WpWnZ_train")
    pandas_to_hdf5(df_val, "original", "WpWnZ_val")
    pandas_to_hdf5(df_test, "original", "WpWnZ_test")

if __name__ == "__main__":
    main()

