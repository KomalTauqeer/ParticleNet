import awkward as ak
import pandas

f = ak.load("btag_test/multitraining_sets/WpWnZ_genmatched_train_UL18_0.awkd")
#df = ak.pandas.df(f)
#print(df)
#print(f["part_deltaR"])
#print(f["part_phirel"])
#print(f["part_etarel"])
#print(f["part_px"])
#print(f["part_charge"])
#print(f["label"])
#print(f["jetcharge"])
#print(f["subjet1charge"])
#print(f["subjet2charge"])

v1 = f['doublebtag']
print(type(v1))
print(v1)
