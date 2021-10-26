import awkward as ak
import pandas

f = ak.load("train_file_0.awkd")
#df = ak.pandas.df(f)
#print(df)
print(f["part_deltaR"])
print(f["part_phirel"])
print(f["part_etarel"])
print(f["part_px"])
print(f["part_charge"])
print(f["label"])
