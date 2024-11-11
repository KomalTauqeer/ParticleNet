# Purpose: Split the input root files into 3 parts train, val, test in ratio 60:20:20. This is to keep track of events used for training and validation.

import sys
import math
from array import array
import numpy as np
import ROOT
import argparse
from constants import *

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Select from UL16preVFP, UL16postVFP, UL17, or UL18", default = 'UL18')
args = parser.parse_args()
year = args.year

def train_test_split(region, sample):
    if 'Data' in sample and year == 'UL18':
        ifile_name = inputfilepath[region][year]+datafilename_UL18[region][sample]
    elif 'Data' in sample and year != 'UL18':
        ifile_name = inputfilepath[region][year]+datafilename[region][sample]
    else:
        ifile_name = inputfilepath[region][year]+inputfilename[region][sample]
    
    print ("***************************** Reading {} *********************************".format(ifile_name))
    ifile = ROOT.TFile.Open(ifile_name, "READ")
    itree = ifile.Get(treename)

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
   
    ifile_name = inputfilepath[region][year]+'/'+systype+'/'+sysdir+'/'+inputfilename[region][sample][:-len('.root')]+'_'+systype+'_'+sysdir+'.root'
  
    print ("***************************** Reading {} *********************************".format(ifile_name))
    ifile = ROOT.TFile.Open(ifile_name, "READ")
    itree = ifile.Get(treename)
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
    
    #train_test_split("ZJetsCR", "ZJets")

    #for sample in data_samples:
    #    train_test_split("TTCR", sample)

    for sample in samples:
    	train_test_split("TTCR", sample)

    for sample in samples:
        for unc in ["jec", "jer"]:
            for uncdir in ["up", "down"]:
                split_sys_files("TTCR", sample, unc, uncdir)


if __name__ == '__main__':
    main()

