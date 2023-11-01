# activate vir env: conda activate tf
import os
import sys
import optparse
import meta_data
import data
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

parser = optparse.OptionParser()
parser.add_option("--year", "--y", dest="year", default= "UL18")
parser.add_option("--sample", "--s", dest="sample", default= "TT")
parser.add_option("--region", "--r", dest="region", default= "TTCR")
parser.add_option("--mode", "--m", dest="mode", default= "train", help='choose from train, test or eval mode of run')
parser.add_option("--outdir", "--odir", dest="outdir", default= "original")

(options,args) = parser.parse_args()
year = options.year
sample = options.sample
region = options.region
outdir = options.outdir
mode = options.mode

filepath = meta_data.inputfilepath[region][year]
filename = meta_data.inputfilename[region][sample]
treename = meta_data.treename
inputvariables = meta_data.variables
labels = meta_data.labels

if mode == 'train' or mode == 'test':
    dataset = data.prepare_input_dataset(filepath, filename, treename, region, inputvariables, labels)
    dataset = shuffle(dataset, random_state=42)
elif mode == 'eval':
    dataset = data.prepare_input_dataset(filepath, filename, treename, region, inputvariables, []) #Fix this right now cannot work for eval mode

print ("Total number of W+: {} ".format(dataset['lep_charge'].value_counts()[-1.0]))
print ("Total number of W-: {} ".format(dataset['lep_charge'].value_counts()[1.0]))

if mode == 'train':
    df_train, df_valandtest = train_test_split(dataset, test_size=0.4)
    df_val, df_test = train_test_split(df_valandtest, test_size=0.5)
    
    print (df_train)
    print (df_val.head(5))
    print (df_test.head(5))
    print ("Number of W+ in train set: {} ".format(df_train['lep_charge'].value_counts()[-1.0]))
    print ("Number of W- in train set: {} ".format(df_train['lep_charge'].value_counts()[1.0]))
    print ("Number of W+ in val set: {} ".format(df_val['lep_charge'].value_counts()[-1.0]))
    print ("Number of W- in val set: {} ".format(df_val['lep_charge'].value_counts()[1.0]))
    print ("Number of W+ in test set: {} ".format(df_test['lep_charge'].value_counts()[-1.0]))
    print ("Number of W- in test set: {} ".format(df_test['lep_charge'].value_counts()[1.0]))

if not os.path.isdir(outdir):
    os.mkdir(outdir)

if mode == 'train':
    data.save_dataset(df_train, outdir, 'Train_{}_{}'.format(sample,year))
    data.save_dataset(df_val, outdir, 'Val_{}_{}'.format(sample,year))
    data.save_dataset(df_test, outdir, 'Test_{}_{}'.format(sample,year))
elif mode == 'test':
    data.save_dataset(dataset, outdir, 'Test_{}_{}_{}'.format(region,sample,year))
elif mode == 'eval':
    print ("No setup for eval mode!")
