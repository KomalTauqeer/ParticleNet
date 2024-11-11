# imports from standard python
from __future__ import print_function
import sys
from array import array
import glob
import shutil
import time
from multiprocessing import Process, Manager

# import from local packages
import ROOT
#ROOT.PyConfig.IgnoreCommandLineOptions = True
from ROOT import TFile, TTree, TH1F

# imports from pip packages
from tqdm import tqdm
import pandas as pd
import numpy as np
#from numpy.lib.recfunctions import stack_arrays
#from root_pandas import read_root, to_root
#from root_numpy import root2array, tree2array, array2tree, array2root, root2rec

def add_branch(filename, treename, branchname, branchtype, data):

    # open input file and tree
    ifile = TFile(filename,'READ')
    itree = ifile.Get(treename)

    ofilename = filename.rstrip('.root') + '_jetchargetagger_WpWn.root'
    # create output file
    ofile = TFile(ofilename,'RECREATE')

    # clone tree, FIX: hardcoded
    #ofile.mkdir('utm')
    #ofile.cd('utm')

    # set branch inactive in itree if it already exists
    if itree.FindBranch(branchname):
        itree.SetBranchStatus(branchname,0)

    # clone itree
    print('--- Cloning input file ...')
    otree = itree.CloneTree()
    otree.Write()

    # make new variable and add it as a branch to the tree
    y_helper = array(branchtype.lower(),[0])
    branch = otree.Branch(branchname, y_helper, branchname + '/' + branchtype)

    # get number of entries and check if size matches the data
    n_entries = otree.GetEntries()
    if n_entries != data.size:
        print('mismatch in input tree entries and new branch entries!')

    # fill the branch
    print('--- Adding branch %s in %s:%s ...' %(branchname, filename, treename))
    for i in tqdm(range(n_entries)):
        otree.GetEntry(i)
        y_helper[0] = data[i]
        branch.Fill()

    # write new branch to the tree
    ofile.Write("",TFile.kOverwrite)

    # close input file
    ifile.Close()

    # close output file
    ofile.Close()

    # overwrite old file
    #print('--- Overwrite original file ...')
    #shutil.move(filename + '.mist', filename)

#def add_branches(filename, treename, branchname1, branchtype1, data1, branchname2, branchtype2, data2):
#
#    # open input file and tree
#    ifile = TFile(filename,'READ')
#    itree = ifile.Get(treename)
#
#    ofilename = filename.rstrip('.root') + '_jetchargetaggerMulti.root'
#    # create output file
#    ofile = TFile(ofilename,'RECREATE')
#
#    # clone tree, FIX: hardcoded
#    #ofile.mkdir('utm')
#    #ofile.cd('utm')
#
#    # set branch inactive in itree if it already exists
#    if itree.FindBranch(branchname1):
#        itree.SetBranchStatus(branchname1,0)
#    if itree.FindBranch(branchname2):
#        itree.SetBranchStatus(branchname2,0)
#    # clone itree
#    print('--- Cloning input file ...')
#    otree = itree.CloneTree()
#    otree.Write()
#
#    # make new variable and add it as a branch to the tree
#    y_helper1 = array(branchtype1.lower(),[0])
#    branch1 = otree.Branch(branchname1, y_helper1, branchname1 + '/' + branchtype1)
#    y_helper2 = array(branchtype2.lower(),[0])
#    branch2 = otree.Branch(branchname2, y_helper2, branchname2 + '/' + branchtype2)
#
#    # get number of entries and check if size matches the data
#    n_entries = otree.GetEntries()
#    print (n_entries)
#    print (data1.size)
#    print (data2.size)
#    if n_entries != data1.size:
#        print('mismatch in input tree entries and new branch1 entries!')
#    if n_entries != data2.size:
#        print('mismatch in input tree entries and new branch2 entries!')
#
#    # fill the branch
#    print('--- Adding branch %s in %s:%s ...' %(branchname1, filename, treename))
#    print('--- Adding branch %s in %s:%s ...' %(branchname2, filename, treename))
#    for i in tqdm(range(n_entries)):
#        otree.GetEntry(i)
#        y_helper1[0] = data1[i]
#        branch1.Fill()
#        y_helper2[0] = data2[i]
#        branch2.Fill()
#
#    # write new branch to the tree
#    ofile.Write("",TFile.kOverwrite)
#
#    # close input file
#    ifile.Close()
#
#    # close output file
#    ofile.Close()

    # overwrite old file
    #print('--- Overwrite original file ...')
    #shutil.move(filename + '.mist', filename)

def add_fourbranches(filename, treename, branchname1, branchtype1, data1, branchname2, branchtype2, data2, branchname3, branchtype3, data3, branchname4, branchtype4, data4):

    # open input file and tree
    ifile = TFile(filename,'READ')
    itree = ifile.Get(treename)

    ofilename = filename.rstrip('.root') + '_jetchargetaggerMulti.root'
    # create output file
    ofile = TFile(ofilename,'RECREATE')

    # clone tree, FIX: hardcoded
    #ofile.mkdir('utm')
    #ofile.cd('utm')

    # set branch inactive in itree if it already exists
    if itree.FindBranch(branchname1):
        itree.SetBranchStatus(branchname1,0)
    if itree.FindBranch(branchname2):
        itree.SetBranchStatus(branchname2,0)
    if itree.FindBranch(branchname3):
        itree.SetBranchStatus(branchname3,0)
    if itree.FindBranch(branchname4):
        itree.SetBranchStatus(branchname4,0)
    # clone itree
    print('--- Cloning input file ...')
    otree = itree.CloneTree()
    otree.Write()

    # make new variable and add it as a branch to the tree
    y_helper1 = array(branchtype1.lower(),[0])
    branch1 = otree.Branch(branchname1, y_helper1, branchname1 + '/' + branchtype1)
    y_helper2 = array(branchtype2.lower(),[0])
    branch2 = otree.Branch(branchname2, y_helper2, branchname2 + '/' + branchtype2)
    y_helper3 = array(branchtype3.lower(),[0])
    branch3 = otree.Branch(branchname3, y_helper3, branchname3 + '/' + branchtype3)
    y_helper4 = array(branchtype4.lower(),[0])
    branch4 = otree.Branch(branchname4, y_helper4, branchname4 + '/' + branchtype4)

    # get number of entries and check if size matches the data
    n_entries = otree.GetEntries()
    print (n_entries)
    print (data1.size)
    print (data2.size)
    print (data3.size)
    print (data4.size)
    if n_entries != data1.size:
        print('mismatch in input tree entries and new branch1 entries!')
    if n_entries != data2.size:
        print('mismatch in input tree entries and new branch2 entries!')
    if n_entries != data3.size:
        print('mismatch in input tree entries and new branch3 entries!')
    if n_entries != data4.size:
        print('mismatch in input tree entries and new branch4 entries!')

    # fill the branch
    print('--- Adding branch %s in %s:%s ...' %(branchname1, filename, treename))
    print('--- Adding branch %s in %s:%s ...' %(branchname2, filename, treename))
    print('--- Adding branch %s in %s:%s ...' %(branchname3, filename, treename))
    print('--- Adding branch %s in %s:%s ...' %(branchname4, filename, treename))
    for i in tqdm(range(n_entries)):
        otree.GetEntry(i)
        y_helper1[0] = data1[i]
        branch1.Fill()
        y_helper2[0] = data2[i]
        branch2.Fill()
        y_helper3[0] = data3[i]
        branch3.Fill()
        y_helper4[0] = data4[i]
        branch4.Fill()

    # write new branch to the tree
    ofile.Write("",TFile.kOverwrite)

    # close input file
    ifile.Close()

    # close output file
    ofile.Close()

    # overwrite old file
    #print('--- Overwrite original file ...')
    #shutil.move(filename + '.mist', filename)

def add_fivebranches(filename, treename, branchname1, branchtype1, data1, branchname2, branchtype2, data2, branchname3, branchtype3, data3, branchname4, branchtype4, data4, branchname5, branchtype5, data5):

    # open input file and tree
    ifile = TFile(filename,'READ')
    itree = ifile.Get(treename)

    ofilename = filename.rstrip('.root') + '_jetchargetaggerMulti5.root'
    # create output file
    ofile = TFile(ofilename,'RECREATE')

    # clone tree, FIX: hardcoded
    #ofile.mkdir('utm')
    #ofile.cd('utm')

    # set branch inactive in itree if it already exists
    if itree.FindBranch(branchname1):
        itree.SetBranchStatus(branchname1,0)
    if itree.FindBranch(branchname2):
        itree.SetBranchStatus(branchname2,0)
    if itree.FindBranch(branchname3):
        itree.SetBranchStatus(branchname3,0)
    if itree.FindBranch(branchname4):
        itree.SetBranchStatus(branchname4,0)
    if itree.FindBranch(branchname5):
        itree.SetBranchStatus(branchname5,0)
    # clone itree
    print('--- Cloning input file ...')
    otree = itree.CloneTree()
    otree.Write()

    # make new variable and add it as a branch to the tree
    y_helper1 = array(branchtype1.lower(),[0])
    branch1 = otree.Branch(branchname1, y_helper1, branchname1 + '/' + branchtype1)
    y_helper2 = array(branchtype2.lower(),[0])
    branch2 = otree.Branch(branchname2, y_helper2, branchname2 + '/' + branchtype2)
    y_helper3 = array(branchtype3.lower(),[0])
    branch3 = otree.Branch(branchname3, y_helper3, branchname3 + '/' + branchtype3)
    y_helper4 = array(branchtype4.lower(),[0])
    branch4 = otree.Branch(branchname4, y_helper4, branchname4 + '/' + branchtype4)
    y_helper5 = array(branchtype5.lower(),[0])
    branch5 = otree.Branch(branchname5, y_helper5, branchname5 + '/' + branchtype5)

    # get number of entries and check if size matches the data
    n_entries = otree.GetEntries()
    print (n_entries)
    print (data1.size)
    print (data2.size)
    print (data3.size)
    print (data4.size)
    print (data5.size)
    if n_entries != data1.size:
        print('mismatch in input tree entries and new branch1 entries!')
    if n_entries != data2.size:
        print('mismatch in input tree entries and new branch2 entries!')
    if n_entries != data3.size:
        print('mismatch in input tree entries and new branch3 entries!')
    if n_entries != data4.size:
        print('mismatch in input tree entries and new branch4 entries!')
    if n_entries != data5.size:
        print('mismatch in input tree entries and new branch4 entries!')

    # fill the branch
    print('--- Adding branch %s in %s:%s ...' %(branchname1, filename, treename))
    print('--- Adding branch %s in %s:%s ...' %(branchname2, filename, treename))
    print('--- Adding branch %s in %s:%s ...' %(branchname3, filename, treename))
    print('--- Adding branch %s in %s:%s ...' %(branchname4, filename, treename))
    print('--- Adding branch %s in %s:%s ...' %(branchname5, filename, treename))
    for i in tqdm(range(n_entries)):
        otree.GetEntry(i)
        y_helper1[0] = data1[i]
        branch1.Fill()
        y_helper2[0] = data2[i]
        branch2.Fill()
        y_helper3[0] = data3[i]
        branch3.Fill()
        y_helper4[0] = data4[i]
        branch4.Fill()
        y_helper5[0] = data5[i]
        branch5.Fill()

    # write new branch to the tree
    ofile.Write("",TFile.kOverwrite)

    # close input file
    ifile.Close()

    # close output file
    ofile.Close()

    # overwrite old file
    #print('--- Overwrite original file ...')
    #shutil.move(filename + '.mist', filename)

