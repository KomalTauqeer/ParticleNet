import os
import sys
import datetime
import optparse
import numpy as np
import awkward
import tensorflow as tf
from tensorflow import keras
from tf_keras_model import get_particle_net, get_particle_net_lite
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

parser = optparse.OptionParser()
parser.add_option("--use_gpu" , "--use_gpu", action="store_true", dest = "gpu_train", help = "gpu training", default = True)
parser.add_option("--gpu_device", type = "int", help = "choose from 0,1,2,3", default= 1)
parser.add_option("--year", "--y", dest="year", default= "UL18")
parser.add_option("--outdir", "--outdir", dest="outdir", default= "")
(options,args) = parser.parse_args()
gpu_training = options.gpu_train
gpu_device = options.gpu_device
year= options.year
outdir = options.outdir

if gpu_training:
    #tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
      try:
        tf.config.set_visible_devices(gpus[gpu_device], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
      except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

def stack_arrays(a, keys, axis=-1):
    flat_arr = np.stack([a[k].flatten() for k in keys], axis=axis)
    return awkward.JaggedArray.fromcounts(a[keys[0]].counts, flat_arr)

def pad_array(a, maxlen, value=0., dtype='float32'):
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = s[:maxlen].astype(dtype)
        x[idx, :len(trunc)] = trunc
    return x

class Dataset(object):

    def __init__(self, filepath, feature_dict = {}, label='label', pad_len=100, data_format='channel_first'):
        self.filepath = filepath
        self.feature_dict = feature_dict
        if len(feature_dict)==0:
            feature_dict['points'] = ['part_etarel', 'part_phirel']
            feature_dict['features'] = ['part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel', 'part_charge', 'part_deltaR']
            feature_dict['add_features'] = ['fatjet_subjet1_btag', 'fatjet_subjet2_btag']
            feature_dict['mask'] = ['part_pt_log']
        self.label = label
        self.pad_len = pad_len
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        self._values = {}
        self._label = None
        self._load()

    def _load(self):
        logging.info('Start loading file %s' % self.filepath)
        counts = None
        with awkward.load(self.filepath) as a:
            self._label = a[self.label]
            for k in self.feature_dict:
                arrs = []
                if not k == 'add_features':
                    cols = self.feature_dict[k]
                    if not isinstance(cols, (list, tuple)):
                        cols = [cols]
                    for col in cols:
                        if counts is None:
                            counts = a[col].counts
                        else:
                            assert np.array_equal(counts, a[col].counts)
                        arrs.append(pad_array(a[col], self.pad_len))
                else:
                    column = self.feature_dict[k]
                    for col in column:
                        arrs.append(a[col])
                self._values[k] = np.stack(arrs, axis=self.stack_axis)
            print(self._values['features'])
            print(self._values['add_features'])
                    
        logging.info('Finished loading file %s' % self.filepath)

    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        if key==self.label:
            return self._label
        else:
            return self._values[key]
    
    @property
    def X(self):
        return self._values
    
    @property
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]

def train_multi():
    #Load training and validation dataset
    #train_dataset = Dataset('preprocessing/converted/multitraining_sets/WpWnZ_genmatched_train_{}_0.awkd'.format(year), data_format='channel_last')
    #val_dataset = Dataset('preprocessing/converted/multitraining_sets/WpWnZ_genmatched_val_{}_0.awkd'.format(year), data_format='channel_last')
    train_dataset = Dataset('preprocessing/converted/btagVars/WpWnZ_genmatched_train_{}_0.awkd'.format(year), data_format='channel_last')
    val_dataset = Dataset('preprocessing/converted/btagVars/WpWnZ_genmatched_val_{}_0.awkd'.format(year), data_format='channel_last')
    
    model_type = 'particle_net_lite' # choose between 'particle_net' and 'particle_net_lite'
    num_classes = train_dataset.y.shape[1]
    input_shapes = {k:train_dataset[k].shape[1:] for k in train_dataset.X}
    
    if 'lite' in model_type:
        model = get_particle_net_lite(num_classes, input_shapes)
    else:
        model = get_particle_net(num_classes, input_shapes)
    
    #Training parameters
    batch_size = 1024 if 'lite' in model_type else 128
    epochs = 30
    
    def lr_schedule(epoch):
        lr = 1e-4
        if epoch > 10:
            lr *= 0.1
        elif epoch > 20:
            lr *= 0.01
        logging.info('Learning rate: %f'%lr)
        return lr
    
    #opt = keras.optimizers.Adam(learning_rate= 1e-5)
    opt = keras.optimizers.Adam(learning_rate= lr_schedule(0))
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer=opt,
                  metrics=['accuracy'])
    
    #model.summary()
    #keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    
    # Prepare model model saving directory.
    save_dir = outdir+ '/model_checkpoints'
    model_name = '%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True)
    
    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    progress_bar = keras.callbacks.ProgbarLogger()
    earlystopping = keras.callbacks.EarlyStopping(verbose=True, patience=10, monitor='val_loss')
    log_dir = outdir + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    callbacks = [checkpoint, lr_scheduler, progress_bar, earlystopping, tensorboard_callback]
    #callbacks = [checkpoint, progress_bar, earlystopping, tensorboard_callback]
    
    train_dataset.shuffle()
    val_dataset.shuffle()
    
    history = model.fit(train_dataset.X, train_dataset.y,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(val_dataset.X, val_dataset.y),
              shuffle=True,
              callbacks=callbacks)
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('{}/accuracy.pdf'.format(outdir))
    plt.clf()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('{}/loss.pdf'.format(outdir))
    plt.close()

def main():
    train_multi()

if __name__ == "__main__":
    main()

