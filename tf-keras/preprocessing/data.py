import uproot4
import pandas

def root2df(sfile, tree_name, variables, **kwargs):
    f = uproot4.open(sfile)
    tree = f[tree_name]
    df = tree.arrays(variables, library="pd")
    return df

def pandas_to_hdf5(df, outdir, outfilename):
    df.to_hdf(outdir + '/'+ outfilename +'.h5', key='table', mode='w')

def save_dataset(dataset, outdir, outfilename):
    pandas_to_hdf5(dataset, outdir, outfilename)

def unstack_multi_df(df):
    df = df.unstack()
    #Remove subentries level and rename dataframe columns 
    df.columns = [a[0] + "_" +str(a[1]) for a in df.columns.to_flat_index()]
    return df

def merge_df(first, second, index_ignore=True):
    df = pd.concat((first, second), ignore_index=index_ignore)
    return df

def prepare_input_dataset(filepath, filename, treename, region, variables, labels):
    dataset = root2df(filepath+filename, treename, variables)
    if region == 'TTCR' or region == 'VBSSR':
        labels = root2df(filepath+filename, treename, labels)
    dataset = unstack_multi_df(dataset)
    if region == 'ZJetsCR':
        dataset['lep_charge'] = np.array(0.)
        plot_dataset['lep_charge'] = np.array(0.) 
    if region == 'TTCR' or region == 'VBSSR': 
        dataset = dataset.join(labels)
    return dataset

