import pandas as pd

#f = pd.read_hdf('../tutorial_datasets/original/train.h5','table')
f1 = pd.read_hdf('Train_TTToSemiLeptonic_2016v3.h5','table')
f2 = pd.read_hdf('Test_TTToSemiLeptonic_2016v3.h5','table')
f3 = pd.read_hdf('Val_TTToSemiLeptonic_2016v3.h5','table')
print(f1)
print(f2)
print(f3)
print(f1.keys())
print(f2.keys())
print(f3.keys())


