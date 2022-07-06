#This script plots PN output score in matplotlib, however I have written another script to plot this via Pyroot. see UHH2/scripts/plot/TTSemiLeptonic/DataVsMC/PlotPNCSV.py

import pandas as pd
import matplotlib.pyplot as plt

PNOutput_TTMC = pd.read_csv('PNOutput_TTMC_with_col.csv').to_numpy()
TrueOutput_TTMC = pd.read_csv('TrueOutput_TTMC_with_col.csv').to_numpy()
PNOutput_SingleMuon = pd.read_csv('PNOutput_SingleMuon_with_col.csv').to_numpy()
TrueOutput_SingleMuon = pd.read_csv('TrueOutput_SingleMuon_with_col.csv').to_numpy()

#print(PNOutput_TTMC)
#print(TrueOutput_TTMC)
#print(PNOutput_SingleMuon)
#print(TrueOutput_SingleMuon)


plt.hist(PNOutput_TTMC[TrueOutput_TTMC[:,0]==1,0],30,histtype='step', density= 'True',color='red',label='$\mathrm{W^+}$')
plt.hist(PNOutput_TTMC[TrueOutput_TTMC[:,0]==0,0],30,histtype='step', density= 'True',color='blue',label='$\mathrm{W^-}$')
np,binsp,patchesp= plt.hist(PNOutput_SingleMuon[TrueOutput_SingleMuon[:,0]==1,0],30, histtype='step', density= 'True',color='red',label='$\mathrm{W^+}$', alpha=0)
nn,binsn,patchesn= plt.hist(PNOutput_SingleMuon[TrueOutput_SingleMuon[:,0]==0,0],30, histtype='step', density= 'True',color='blue',label='$\mathrm{W^-}$', alpha=0)
plt.scatter(binsp[:-1]+ 0.5*(binsp[1:] - binsp[:-1]), np, marker='o', c='red', s=40, alpha=1)
plt.scatter(binsn[:-1]+ 0.5*(binsn[1:] - binsn[:-1]), nn, marker='o', c='blue', s=40, alpha=1)
plt.legend(loc='upper right')
plt.ylabel('Events')
plt.xlabel('Particle Net score')
plt.savefig('test.svg')
plt.close()

