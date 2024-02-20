#Purpose: Apply preselections on the tuples based on btag disc values of the subjets of the fatjet to define two regions: b-tagged and non-btagged.
#Afterwards we will do the multi classification in both these categories

import sys
import math
from array import array
import numpy as np
import ROOT
import argparse
from meta_data import *

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Select from UL16preVFP, UL16postVFP, UL17, or UL18", default="UL18")
args = parser.parse_args()
year = args.year

samples = ["TT", "ZJets"]
sample_region = {"TT": "TTCR", "ZJets": "ZJetsCR"}
input_treename = "AnalysisTree"

def apply_preselection(category):
    for sample in samples:
        ifile_name = inputfilepath[sample_region[sample]][year]+inputfilename[sample_region[sample]][sample]
        ifile = ROOT.TFile.Open(ifile_name, "READ")
        itree = ifile.Get(input_treename)
        nentries = int(itree.GetEntries())
        print ('sample name is {} and file name is {}'.format(sample, ifile_name))
        print (nentries)
  
        subjet1_btag = array('f', [0])
        itree.SetBranchAddress('fatjet_subjet1_btag_DeepFlavour_b', subjet1_btag)

        ofile_name = (ifile_name.rstrip('.root')) +  '_{}_region.root'.format(category)
        ofile = ROOT.TFile.Open(ofile_name, "RECREATE")
        print('--- Cloning input tree header ...')
        otree = itree.CloneTree(0)
        
        for ientry in range(nentries):
            print (ientry)
            itree.GetEntry(ientry)
            if category == "btagged":
                if (subjet1_btag[0] > 0.6):
                    otree.Fill()
            if category == "nonbtagged":
                if (subjet1_btag[0] < 0.6) and (subjet1_btag[0] > 0.1):
                    otree.Fill()

        otree.Write()
        ofile.Close()
        ifile.Close()

def main():
    apply_preselection("btagged")
    apply_preselection("nonbtagged")


if __name__ == '__main__':
    main()

