# Purpose: Split the input root files into 3 parts ABC to be used while training SD-DNN

import sys
import math
from array import array
import numpy as np
import ROOT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Select from UL16preVFP, UL16postVFP, UL17, or UL18", default = 'UL18')
args = parser.parse_args()

year = args.year

samples = ["TT", "Data_singlemuon", "Data_singleelectron"]

inputPathPrefix = {
                    "UL16preVFP": "/ceph/ktauqeer/ULNtuples/UL16preVFP/TTCR/" ,
                    "UL16postVFP": "/ceph/ktauqeer/ULNtuples/UL16postVFP/TTCR/",
                    "UL17": "/ceph/ktauqeer/ULNtuples/UL17/TTCR/",
                    "UL18": "/ceph/ktauqeer/ULNtuples/UL18/TTCR/",
                  }

inputFileName = {
                "TT": "TTCR_TTToSemiLeptonic.root",
                }

dataFileName = {
                "Data_singlemuon": "TTCR_SingleMuon_combined.root",
                "Data_singleelectron": "TTCR_SingleElectron_combined.root",

               } 

dataFileName_UL18 = {
                "Data_singlemuon": "TTCR_SingleMuon_combined.root",
                "Data_singleelectron": "TTCR_EGamma_combined.root",

               } 

input_treename = "AnalysisTree"

def train_test_split(sample):

   for part in ['train', 'val', 'test']:
       if 'Data' in sample and year == 'UL18':
           ifile_name = inputPathPrefix[year]+dataFileName_UL18[sample]
       elif 'Data' in sample and year != 'UL18':
           ifile_name = inputPathPrefix[year]+dataFileName[sample]
       else:
           ifile_name = inputPathPrefix[year]+inputFileName[sample]
  
       print (ifile_name)
       ifile = ROOT.TFile.Open(ifile_name, "READ")
       itree = ifile.Get(input_treename)

       nentries = int(itree.GetEntries())
       print (nentries)
       eventid = array('i', [0])
       itree.SetBranchAddress('event', eventid)

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

def split_sys_files(sample, systype, sysdir):
   
    for part in ['train', 'val', 'test']:
       ifile_name = inputPathPrefix[year]+'/'+systype+'/'+sysdir+'/'+inputFileName[sample][:-len('.root')]+'_'+systype+'_'+sysdir+'.root'
  
       print (ifile_name)
       ifile = ROOT.TFile.Open(ifile_name, "READ")
       itree = ifile.Get(input_treename)

       nentries = int(itree.GetEntries())
       print (nentries)
       eventid = array('i', [0])
       itree.SetBranchAddress('event', eventid)

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
    #for sample in samples:

    #    train_test_split(sample) #This will split rootfile into three parts train, val, test (60, 20, 20)

    #Redo TT after top pt weights
    train_test_split("TT") 
    for unc in ["jec", "jer"]:
        for uncdir in ["up", "down"]:
            split_sys_files("TT", unc, uncdir)

if __name__ == '__main__':
    main()

