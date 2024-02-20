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
parser.add_option("--use_gpu" , "--use_gpu", action="store_true", dest = "gpu_train", help = "gpu training", default = False)
parser.add_option("--gpu_device", type = "int", help = "choose from 0,1,2,3", default= 2)
parser.add_option("--year", "--y", dest="year", help = "UL16preVFP, UL16postVFP, UL17, UL18", default= "UL18")
parser.add_option("--outdir", "--outdir", dest="outdir", help = "Set a flag for the outdir of training results", default= None)
(options,args) = parser.parse_args()
gpu_training = options.gpu_train
gpu_device = options.gpu_device
year= options.year
dirflag = options.outdir

if dirflag is None:
    print ("Please specify an outdir flag to distinguish the new training results from the past ones.")
    sys.exit()

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

def train():
    print ("Start loading the training and validation dataset .....")
    train_dataset = Dataset('preprocessing/converted/binary_training/{y}/WpWn_train_{y}_0.awkd'.format(y=year), data_format='channel_last')
    val_dataset = Dataset('preprocessing/converted/binary_training/{y}/WpWn_val_{y}_0.awkd'.format(y=year), data_format='channel_last')
    model_type = 'particle_net_lite' # choose between 'particle_net' and 'particle_net_lite'
    print ("Using model {}".format(model_type))
    num_classes = train_dataset.y.shape[1]
    print ("Number of output nodes: {}".format(num_classes))
    input_shapes = {k:train_dataset[k].shape[1:] for k in train_dataset.X}
    print ("Number of inputs: {}".format(input_shapes))
    
    if 'lite' in model_type:
        model = get_particle_net_lite(num_classes, input_shapes)
    else:
        model = get_particle_net(num_classes, input_shapes)
    
    # Training parameters
    batch_size = 1024 if 'lite' in model_type else 384
    epochs = 30
    print ("Hyper parameters: \n Epochs: {} \n batch_size: {} \n".format(epochs,batch_size))

    def lr_schedule(epoch):
        lr = 1e-3
        if epoch > 10:
            lr *= 0.1
        elif epoch > 20:
            lr *= 0.01
        logging.info('Learning rate: %f'%lr)
        return lr

    model.compile(loss='binary_crossentropy', 
                  #optimizer=keras.optimizers.Adam(learning_rate=0.00001),
                  optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
                  metrics=['accuracy'])
    #model.summary()
    #keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

    # Prepare model model saving directory.
    #save_dir = 'training_results_Oct23_lrsch_1e-3/model_checkpoints'
    save_dir = 'binary_training/{}/model_checkpoints'.format(dirflag)
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
    log_dir = "binary_training/{}/logs/fit/".format(dirflag) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [checkpoint, lr_scheduler, progress_bar, tensorboard_callback]
    
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
    plt.savefig('binary_training/{}/accuracy.pdf'.format(dirflag))
    plt.clf()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('binary_training/{}/loss.pdf'.format(dirflag))
    plt.close()

def main():
    train()

if __name__ == "__main__":
    main()
