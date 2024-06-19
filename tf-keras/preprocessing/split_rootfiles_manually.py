# Purpose: Split the input root files into 3 parts train, val, test in ratio 60:20:20. This is to keep track of events used for training and validation.

import sys
import math
from array import array
import numpy as np
import ROOT
import argparse
import meta_data_new

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Select from UL16preVFP, UL16postVFP, UL17, or UL18", default = 'UL18')
parser.add_argument("--mode" , "--mode", dest = "mode", help = "binary/ternary/eval", default = None)
parser.add_argument("--region", "--r", dest="region", help = "TTCR, VBSSR, ZJetsCR", default= None)
parser.add_argument("--sample", "--s", dest="sample", help = "Name of the sample, see meta_data_new.py", default= None)
args = parser.parse_args()
year = args.year
mode = args.mode
region = args.region
sample = args.sample

def train_test_split(region, sample, year):
    inputPathPrefix = meta_data_new.inputfilepath
    inputFileName = meta_data_new.inputfilename
    dataFileName = meta_data_new.datafilename
    dataFileName_UL18 = meta_data_new.datafilename_UL18
    input_treename = meta_data_new.treename

    if 'Data' in sample and year == 'UL18':
        ifile_name = inputPathPrefix[region][year]+dataFileName_UL18[region][sample]
    elif 'Data' in sample and year != 'UL18':
        ifile_name = inputPathPrefix[region][year]+dataFileName[region][sample]
    else:
        ifile_name = inputPathPrefix[region][year]+inputFileName[region][sample]
    
    print (ifile_name)
    ifile = ROOT.TFile.Open(ifile_name, "READ")
    itree = ifile.Get(input_treename)

    nentries = int(itree.GetEntries())
    print (nentries)
    eventid = array('i', [0])
    itree.SetBranchAddress('event', eventid)
    
    for part in ['train', 'val', 'test']:
        ofile_name = (ifile_name.rstrip('.root')) +  '_' + part + '.root'
        ofile = ROOT.TFile.Open(ofile_name, "RECREATE")
        print ('---- Creating {} file ---'.format (part))
        print('--- Cloning input tree header ...')
        otree = itree.CloneTree(0)
        for ientry in range(nentries):
            itree.GetEntry(ientry)
            if (part == 'train'):
                if (eventid[0] % 5 == 0 or eventid[0] % 5 == 1 or eventid[0] % 5 == 2):
                    otree.Fill()
            if (part == 'val'):
                if (eventid[0] % 5 == 3):
                    otree.Fill()
            if (part == 'test'):
                if (eventid[0] % 5 == 4):
                    otree.Fill()

        otree.Write()
        ofile.Close()
    ifile.Close()

def split_sys_files(region, sample, systype, sysdir):
    inputPathPrefix = meta_data_new.inputfilepath
    inputFileName = meta_data_new.inputfilename
    input_treename = meta_data_new.treename
   
    ifile_name = inputPathPrefix[region][year]+'/'+systype+'/'+sysdir+'/'+inputFileName[region][sample][:-len('.root')]+'_'+systype+'_'+sysdir+'.root'
  
    print (ifile_name)
    ifile = ROOT.TFile.Open(ifile_name, "READ")
    itree = ifile.Get(input_treename)
    nentries = int(itree.GetEntries())
    print (nentries)
    eventid = array('i', [0])
    itree.SetBranchAddress('event', eventid)

    for part in ['train', 'val', 'test']:
        ofile_name = (ifile_name.rstrip('.root')) +  '_' + part + '.root'
        ofile = ROOT.TFile.Open(ofile_name, "RECREATE")
        print ('---- Creating {} file ---'.format (part))
        print('--- Cloning input tree header ...')
        otree = itree.CloneTree(0)
        for ientry in range(nentries):
            itree.GetEntry(ientry)
            if (part == 'train'):
                if (eventid[0] % 5 == 0 or eventid[0] % 5 == 1 or eventid[0] % 5 == 2):
                    otree.Fill()
            if (part == 'val'):
                if (eventid[0] % 5 == 3):
                    otree.Fill()
            if (part == 'test'):
                if (eventid[0] % 5 == 4):
                    otree.Fill()

        otree.Write()
        ofile.Close()
    ifile.Close()

def main():
    
    #if mode == "binary": train_test_split("TTCR", "TT", mode) #This will split rootfile into three parts train, val, test (60, 20, 20)
    #elif mode == "ternary": 
    #    train_test_split("TTCR", "TT", mode) 
    #    train_test_split("ZJetsCR", "ZJets", mode)
    #elif mode == "eval":
    #region = "TTCR"
    #samples = ["WJet", "ST", "QCD", "Data_singlemuon", "Data_singleelectron"]
    #for sample in samples:
    #train_test_split("ZJetsCR", "ZJets", year)
    #train_test_split("TTCR", "WJet", year)
    #train_test_split("TTCR", "ST", year)
    #train_test_split("TTCR", "QCD", year)

    for unc in ["jec", "jer"]:
        for uncdir in ["up", "down"]:
            #split_sys_files("TTCR","TT", unc, uncdir)
            split_sys_files("TTCR","WJet", unc, uncdir)
            split_sys_files("TTCR","ST", unc, uncdir)
            split_sys_files("TTCR","QCD", unc, uncdir)

if __name__ == '__main__':
    main()

