import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from ROOT import TFile, TTree, TH1F

def add_branch(filename, treename, branchname, branchtype, data):

    # open input file and tree
    ifile = TFile(filename,'READ')
    itree = ifile.Get(treename)

    # create output file
    ofile = TFile(filename+'_dnn.root','RECREATE')

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

