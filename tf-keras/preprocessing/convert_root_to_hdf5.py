# activate vir env: conda activate tf

import os
import sys
import uproot4
import awkward as ak
import pandas 
import numpy as np
from sklearn.model_selection import train_test_split

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


#Read root file with uproot

#TTSemiLep MC file
#file = uproot4.open('/ceph/ktauqeer/TTSemiLeptonic/2016/RecoNtuples/uhh2.AnalysisModuleRunner.MC.TTToSemiLeptonic_2016v3_PFvars_PxPyPzEQ.root')
file = uproot4.open('/ceph/ktauqeer/ULNtuples/UL16postVFP/TTCR/TTCR_TTToSemiLeptonic.root')
#file = uproot4.open('/ceph/ktauqeer/ULNtuples/UL16postVFP/VBSSR/VBSSR_WZ_combined_v1.root')
#file = uproot4.open('/ceph/ktauqeer/ULNtuples/UL16postVFP/VBSSR/VBSSR_osWW_combined_v1.root')
#file = uproot4.open('/ceph/ktauqeer/ULNtuples/UL16postVFP/VBSSR/VBSSR_ssWW_combined_v1.root')

#TTSemiLep Data file
#file = uproot4.open('/ceph/ktauqeer/TTSemiLeptonic/2016/condorDataNtuples/uhh2.AnalysisModuleRunner.Data.SingleMuon_2016_JetConstits_Pt200_condor.root')

#ssWW VBS files
#file = uproot4.open('/ceph/ktauqeer/WWPolarzation/2016/RecoNtuples/WplusToLNuWplusTo2JJJ/uhh2.AnalysisModuleRunner.MC.WplusToLNuWplusTo2JJJ_EWK_2016v3.root')
#file1 = uproot4.open('/ceph/ktauqeer/WWPolarzation/2016/RecoNtuples/WminusToLNuWminusTo2JJJ/uhh2.AnalysisModuleRunner.MC.WminusToLNuWminusTo2JJJ_EWK_2016v3.root')

#Check file contents
#print (file.classnames())

#Access the tree
AnalysisTree = file['AnalysisTree']
#AnalysisTree1 = file1['AnalysisTree']

#Show tree contents
#AnalysisTree.show()
#AnalysisTree1.show()

#Read TBranches in pandas dataframe
data = AnalysisTree.arrays(["PF_Px", "PF_Py", "PF_Pz", "PF_E", "PF_q"], library = "pd") 
#data1 = AnalysisTree1.arrays(["PF_Px", "PF_Py", "PF_Pz", "PF_E", "PF_q"], library = "pd") 
#FOR TTSEMILEP: data1 = AnalysisTree.arrays(["fatjet_subjet1_charge_k0.5","fatjet_subjet2_charge_k0.5","fatjet_charge_k0.5"],library = "pd")
#FOR TTSEMILEP: 
labels = AnalysisTree.arrays(["lep_charge"],library = "pd")
#FOR sswwVBS: labels = AnalysisTree.arrays(["lep_charge"],library = "pd")
#FOR sswwVBS: labels1 = AnalysisTree1.arrays(["lep_charge"],library = "pd")

#Unstack Multi-Level structure
data = data.unstack()
#data1 = data1.unstack()

#data.columns = data.columns.get_level_values(0)

#Remove subentries level and rename dataframe columns to match those used in ParticleNet tutorial dataset
data.columns = [a[0] + "_" +str(a[1]) for a in data.columns.to_flat_index()]
#data1.columns = [a[0] + "_" +str(a[1]) for a in data1.columns.to_flat_index()]

#FOR TTSEMILEP if you want to add jetcharge vars: data = data.join(data1)
data = data.join(labels)
#data1 = data1.join(labels1)
# For ssww VBS: final_data = pandas.concat([data,data1],ignore_index = True) #combine the data from both files W+ and W-

# For ssww VBS: print (final_data)
# For ssww VBS: print(final_data.loc[:,'lep_charge'])
print (data)
print (data.loc[:,'lep_charge'])


#Splitting of the dataset for training

df_train, df_val, df_test = split_stratified_into_train_val_test(data, stratify_colname='lep_charge', frac_train=0.60, frac_val=0.20, frac_test=0.20)
#df_train, df_val, df_test = split_stratified_into_train_val_test(final_data, stratify_colname='lep_charge', frac_train=0.60, frac_val=0.20, frac_test=0.20)
print (df_train.head(5))
print (df_val.head(5))

#Save the data in .h5 file
#df_train.to_hdf('original/Train_TTToSemiLeptonic_2016v3.h5', key='table', mode='w')
#df_test.to_hdf('original/Test_TTToSemiLeptonic_2016v3.h5', key='table', mode='w')
#df_val.to_hdf('original/Val_TTToSemiLeptonic_2016v3.h5', key='table', mode='w')
#df_train.to_hdf('original/Train_ssWWVBS_2016v3.h5', key='table', mode='w')
#final_data.to_hdf('original/Test_ssWWVBS_2016v3.h5', key='table', mode='w')
#df_val.to_hdf('original/Val_ssWWVBS_2016v3.h5', key='table', mode='w')
#data.to_hdf('original/Test_SingleMuon_2016.h5', key='table', mode='w')
df_train.to_hdf('original/Train_TTToSemiLeptonic_UL16postVFP.h5', key='table', mode='w')
df_test.to_hdf('original/Test_TTToSemiLeptonic_UL16postVFP.h5', key='table', mode='w')
df_val.to_hdf('original/Val_TTToSemiLeptonic_UL16postVFP.h5', key='table', mode='w')
#data.to_hdf('original/VBSSR_ssWW_UL16postVFP.h5', key='table', mode='w')
#data.to_hdf('original/VBSSR_osWW_UL16postVFP.h5', key='table', mode='w')
#data.to_hdf('original/VBSSR_WZ_UL16postVFP.h5', key='table', mode='w')
