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
from load_datasets import *

parser = optparse.OptionParser()
parser.add_option("--use_gpu" , "--use_gpu", action="store_true", dest = "gpu_train", help = "gpu training", default = True)
parser.add_option("--gpu_device", type = "int", help = "choose from 0,1,2,3", default= 1)
parser.add_option("--year", "--y", dest="year", help = "UL16preVFP, UL16postVFP, UL17, UL18", default= "UL18")
(options,args) = parser.parse_args()
gpu_training = options.gpu_train
gpu_device = options.gpu_device
year = options.year

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

def train_multi():

    #Load training and validation dataset
    train_dataset = Dataset('preprocessing/ternary_training/{y}/converted/WpWnZ_train_{y}_0.awkd'.format(y=year), data_format='channel_last')
    val_dataset = Dataset('preprocessing/ternary_training/{y}/converted/WpWnZ_val_{y}_0.awkd'.format(y=year), data_format='channel_last')
    
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
    save_dir = 'ternary_training/{y}/model_checkpoints'.format(y=year)
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
    log_dir = "ternary_training/{y}/logs/fit/".format(y=year) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    callbacks = [checkpoint, lr_scheduler, progress_bar, earlystopping, tensorboard_callback]
    
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
    plt.savefig('ternary_training/{y}/accuracy.pdf'.format(y=year))
    plt.clf()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('ternary_training/{y}/loss.pdf'.format(y=year))
    plt.close()

def main():
    global year
    train_multi()

if __name__ == "__main__":
    main()

