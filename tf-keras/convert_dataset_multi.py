import os
import optparse
import pandas as pd
import numpy as np
import awkward
import uproot_methods
#from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')

parser = optparse.OptionParser()
parser.add_option("--train" , "--train", action="store_true", dest = "do_train", help = "train mode", default = False)
parser.add_option("--eval" , "--eval", action="store_true", dest = "do_eval", help = "eval mode", default = False)
parser.add_option("--year", "--y", dest="year", default= "UL18")
parser.add_option("--region", "--r", dest="region", default= None)
parser.add_option("--sample", "--s", dest="sample", default= None)
parser.add_option("--outdir", "--odir", dest="outdir", default= "converted")
parser.add_option("--srcdir", "--sdir", dest="srcdir", default= "original")
(options,args) = parser.parse_args()
year = options.year
region = options.region
sample = options.sample
outdir = options.outdir
srcdir = options.srcdir


def _transform(dataframe, start=0, stop=-1, jet_size=0.8):
    from collections import OrderedDict
    v = OrderedDict()

    df = dataframe.iloc[start:stop]
    def _col_list(prefix, max_particles=77):
        return ['%s_%d'%(prefix,i) for i in range(max_particles)]

    _px = df[_col_list('PF_Px')].values
    _py = df[_col_list('PF_Py')].values
    _pz = df[_col_list('PF_Pz')].values
    _e = df[_col_list('PF_E')].values
    _q = df[_col_list('PF_q')].values

    mask = _e>0
    n_particles = np.sum(mask, axis=1)

    px = awkward.JaggedArray.fromcounts(n_particles, _px[mask])
    py = awkward.JaggedArray.fromcounts(n_particles, _py[mask])
    pz = awkward.JaggedArray.fromcounts(n_particles, _pz[mask])
    energy = awkward.JaggedArray.fromcounts(n_particles, _e[mask])
    charge = awkward.JaggedArray.fromcounts(n_particles, _q[mask])

    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(px, py, pz, energy)
    pt = p4.pt

    jet_p4 = p4.sum()

    # outputs
    #if options.do_train:
    fatjet_subjet1_btag = df['fatjet_subjet1_btag_DeepFlavour_b'].values
    fatjet_subjet2_btag = df['fatjet_subjet2_btag_DeepFlavour_b'].values
    filtered_fatjet_subjet1_btag = []
    filtered_fatjet_subjet2_btag = []
    for itr in fatjet_subjet1_btag:
        if itr == -80:  filtered_fatjet_subjet1_btag.append(0.)
        else: filtered_fatjet_subjet1_btag.append(itr)
    for itr in fatjet_subjet2_btag:
        if itr == -80:  filtered_fatjet_subjet2_btag.append(0.)
        else: filtered_fatjet_subjet2_btag.append(itr)
    v['fatjet_subjet1_btag'] = np.array(filtered_fatjet_subjet1_btag)
    v['fatjet_subjet2_btag'] = np.array(filtered_fatjet_subjet2_btag)
    old_label = df['lep_charge']
    print (np.count_nonzero(old_label==0.0))
    print (np.count_nonzero(old_label==1.0))
    print (np.count_nonzero(old_label==-1.0))
    new_label = []
    for i in old_label:
        if i == 1: new_label.append([1,0,0])
        if i == -1:new_label.append([0,1,0])
        if i == 0: new_label.append( [0,0,1])
    v['label'] = np.array(new_label)
    v['jet_pt'] = jet_p4.pt
    v['jet_eta'] = jet_p4.eta
    v['jet_phi'] = jet_p4.phi
    v['jet_mass'] = jet_p4.mass
    v['n_parts'] = n_particles

    v['part_px'] = px
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy
    v['part_charge'] = charge

    v['part_pt_log'] = np.log(pt)
    v['part_ptrel'] = pt/v['jet_pt']
    v['part_logptrel'] = np.log(v['part_ptrel'])

    v['part_e_log'] = np.log(energy)
    v['part_erel'] = energy/jet_p4.energy
    v['part_logerel'] = np.log(v['part_erel'])

    v['part_raw_etarel'] = (p4.eta - v['jet_eta'])
    _jet_etasign = np.sign(v['jet_eta'])
    _jet_etasign[_jet_etasign==0] = 1
    v['part_etarel'] = v['part_raw_etarel'] * _jet_etasign

    v['part_phirel'] = p4.delta_phi(jet_p4)
    v['part_deltaR'] = np.hypot(v['part_etarel'], v['part_phirel'])

    def _make_image(var_img, rec, n_pixels = 64, img_ranges = [[-0.8, 0.8], [-0.8, 0.8]]):
        wgt = rec[var_img]
        x = rec['part_etarel']
        y = rec['part_phirel']
        img = np.zeros(shape=(len(wgt), n_pixels, n_pixels))
        for i in range(len(wgt)):
            hist2d, xedges, yedges = np.histogram2d(x[i], y[i], bins=[n_pixels, n_pixels], range=img_ranges, weights=wgt[i])
            img[i] = hist2d
        return img

#     v['img'] = _make_image('part_ptrel', v)

    return v

def convert(source, destdir, basename, step=None, limit=None):
    df = pd.read_hdf(source, key='table')
    logging.info('Total events: %s' % str(df.shape[0]))
    if limit is not None:
        df = df.iloc[0:limit]
        logging.info('Restricting to the first %s events:' % str(df.shape[0]))
    if step is None:
        step = df.shape[0]
    idx=-1
    while True:
        idx+=1
        start=idx*step
        if start>=df.shape[0]: break
        if not os.path.exists(destdir):
            os.makedirs(destdir)
        output = os.path.join(destdir, '%s_%d.awkd'%(basename, idx))
        logging.info(output)
        if os.path.exists(output):
            logging.warning('... file already exist: continue ...')
            continue
        v=_transform(df, start=start, stop=start+step)
        awkward.save(output, v, mode='x')

def main():
    srcDir = '/work/ktauqeer/ParticleNet_multi/tf-keras/preprocessing/' + srcdir
    destDir = '/work/ktauqeer/ParticleNet_multi/tf-keras/preprocessing/' + outdir
    if options.do_train:    
        convert(os.path.join(srcDir, 'WpWnZ_train_{}.h5'.format(year)), destdir=destDir, basename='WpWnZ_genmatched_train_{}'.format(year))
        convert(os.path.join(srcDir, 'WpWnZ_val_{}.h5'.format(year)), destdir=destDir, basename='WpWnZ_genmatched_val_{}'.format(year))
        convert(os.path.join(srcDir, 'WpWnZ_test_{}.h5'.format(year)), destdir=destDir, basename='WpWnZ_genmatched_test_{}'.format(year))
        print ("***Successfully converted training sets***")
    if options.do_eval and options.sample is not None and options.region is not None:
        print ("Converting {}/{}_{}_{}_eval.h5 .................".format(srcDir, region, sample, year))
        convert(os.path.join(srcDir, '{}_{}_{}_eval.h5'.format(region, sample, year)), destdir=destDir, basename='{}_{}_{}'.format(region, sample, year))
        print ("***Successfully converted eval sets***")
    elif (options.sample is None or options.region is None) and options.do_eval:
        print ("Please enter region and sample of the eval dataset")

if __name__ == '__main__':
    main()
