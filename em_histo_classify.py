# Copyright (c) 2018, Science and Technology Facilities Council
# This software is distributed under a BSD licence. See LICENSE.txt.

# Python 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras import initializers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model, to_categorical
from keras.applications import vgg16, inception_v3, resnet50, mobilenet

import h5py

import os
import numpy as np
from numpy import *
from PIL import Image
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import innvestigate
import innvestigate.utils as iutils

# index file for output slices with histograms
index_file = 'EM_slices_blurred_1axis_histo.idx'
# output predictions file
predout = 'predictions.txt'
# file for output weights
weights_out = 'model.hdf5'

# balance classes in training data
balance_dataset = True

# model to use
model_choice = 'hist1'

# load weights from previous run?
load_weights = False
weights_file = 'model.hdf5'
# map output predictions back to original volume, using segmentation files
predictions_to_volume = False
# run iNNvestigate for analysis
run_innvestigate = False

### parameters for training
# perhaps batch size should be a fraction of training set size? affects smoothness of
# convergence
batch_size = 32
epochs = 5
learning_rate = 0.001

class inputHistos():
    """
       Input histograms for machine learning. Training and test data.
    """
    def __init__(self):
        self.num_samples = 10000
        self.histo_size = 10
        # control whether data selection / shuffling is reproducible
        self.random_seed_data = None
 
    def balanceTrainingHistos(self,idx_file, idx_file_new):
        """
        Input raw index file, and output balanced one.
        This assumes there are more "bad" images than "good" ones, and
        a matching number of "bad" ones are selected randomly.
        """
        indexfile = open(idx_file,'r').read().split("\n")
        indexfile.pop()
        listing = []
        for i in indexfile:
            i = i.split(" ")
            listing.append(i)
        bad_images = listing[:]
        good_images = []
        counter = 0
        for image in listing:
            if image[0].startswith("p"):
                good_images.append(image)
                bad_images.remove(image)
                counter+=1

        np.random.seed(self.random_seed_data)
        newarray = np.random.choice(range(0,len(bad_images)),counter)
        bad_random = []
        for i in newarray:
            bad_random.append(bad_images[i])
        newlisting = good_images + bad_random
        self.num_samples = len(newlisting)
        newindex = open(idx_file_new,"w+")
        for i in newlisting:
            newindex.write(" ".join(i)+"\n")
        newindex.close()

    def inputTrainingHistos(self,idx_file):

        ### 2nd example
        indexfile = open(idx_file,'r')
        print('Opened index file ',idx_file)

        # read in images, and flatten each image
        # haven't set dtype
        hist = zeros((self.num_samples,self.histo_size),dtype=int)
        label = zeros(self.num_samples)
        source_pos = zeros((self.num_samples,6),dtype=int)
        i = 0
        n_good = 0
        n_bad = 0

        line = indexfile.readline()
        while line != "":
            words = line.split()
            im = words[0]
            hist[i] = words[8:8+self.histo_size]
            if im[0] == 'p':
                label[i] = 1
                n_good += 1
            elif im[0] == 'b':
                label[i] = 0
                n_bad += 1
            source_pos[i] = (words[2],words[3],words[4],words[5],words[6],words[7])

            i += 1
            if i % 1000 == 0:
                print(i," images read so far")
            line = indexfile.readline()

        indexfile.close()
        print("Histograms read in, of which ",n_good," labelled good and ",n_bad, "labelled bad.")

        # from sklearn, shuffles training data
        hist,label,source_pos = shuffle(hist, label, source_pos, random_state=self.random_seed_data)
        print('input xdata shape after shuffle: ', hist.shape)
        print('input ydata shape after shuffle: ', label.shape)

        # reshape, the images were flattened above
        n_train = int(0.8*self.num_samples)
        print('Using ',n_train,' images for training')
        x_train = hist[:n_train]
        y_train = to_categorical(label[:n_train])
        print('x_train shape after reshape: ', x_train.shape) 
        print('y_train shape after reshape: ', y_train.shape)

        n_test = self.num_samples - n_train
        print('Using ',n_test,' images for testing')
        x_test = hist[n_train:]
        y_test = to_categorical(label[n_train:])
        source_pos_test = source_pos[n_train:]
        print('x_test shape after reshape: ', x_test.shape)
        print('y_test shape after reshape: ', y_test.shape)
        print('source_pos_test shape after reshape: ', source_pos_test.shape)

        return x_train, y_train, x_test, y_test, source_pos_test

class mapMLModel(Sequential):
    """
       Our multi-layer model for maps
    """

    def __init__(self):
        Sequential.__init__(self)
 
    def createHist1(self,input_shape):
        self.add(Dense(80, kernel_initializer=initializers.he_normal(), activation='relu', name='fc1', input_shape=(input_shape,)))
        #self.add(Dropout(0.5))
        self.add(Dense(2, activation='softmax', name='predictions'))

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.accuracy.append(logs.get('acc'))
        print("ugg", self.accuracy)

class WeightsCheck(keras.callbacks.Callback):
    def on_epoch_begin(self, batch, logs={}):
        for layer in model.layers[16:17]:
            print(layer.get_weights())
    def on_epoch_end(self, batch, logs={}):
        for layer in model.layers[16:17]:
            print(layer.get_weights())



###### start here ######

histos = inputHistos()
if balance_dataset:
    histos.balanceTrainingHistos(index_file,"temp_index_file.idx")
    x_train,y_train,x_test,y_test,pos_test = histos.inputTrainingHistos("temp_index_file.idx")
else:
    x_train,y_train,x_test,y_test,pos_test = histos.inputTrainingHistos(index_file)
print("*** Histograms read in ***")

print(x_train[123])
print(y_train[123])
print(x_test[123])
print(y_test[123])

### Define the NN model, based on shape of images
# look at histo features e.g. Shannon entropy
model = mapMLModel()
if model_choice == 'hist1':
    nhist = len(x_train[0])
    model.createHist1(nhist)
else:
    print("Unrecognised model, stopping ...")
    exit(1)

#plot_model(model, to_file='model.png')
for layer in model.layers:
    print("Layer ...")
    print("  ",layer.name)
    print("  ",layer.input_shape)
    print("  ",layer.output_shape)

# initiate RMSprop optimizer
opt = keras.optimizers.SGD(lr=learning_rate)
#opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Let's train the model 
# "loss" is used in training
# "metrics" is used to judge
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
print('Model compiled')
if load_weights:
    model.load_weights(weights_file)
    print('Weights loaded')

print(x_train.shape, y_train.shape)
json_string = model.to_json()
outfile = open('model.json','w')
outfile.write(json_string)
outfile.close()
yaml_string = model.to_yaml()
outfile2 = open('model.yaml','w')
outfile2.write(yaml_string)
outfile2.close()

logging_callback = AccuracyHistory()
weights_check = WeightsCheck()
# or augment images first, see e.g. example 7 of
# https://www.programcreek.com/python/example/89221/keras.preprocessing.image.ImageDataGenerator
if epochs != 0:
    model.fit(x_train[:1000], y_train[:1000],
          batch_size=batch_size,
          epochs=epochs,
          callbacks = [])
    print('Model trained')
    model.save_weights(weights_out)

### Now test
prediction = model.predict(x_test,batch_size=batch_size)

#iNNvestigate
if run_innvestigate:
    pre_sm_model = Model(inputs=model.input,
                     outputs=model.get_layer('fc1').output)
    analyzer = innvestigate.create_analyzer("guided_backprop",pre_sm_model)

fileout = predout
outfile = open(fileout,'w')
if not os.path.isdir('raw_predictions'):
  print("Creating raw_predictions directory ...")
  os.mkdir('raw_predictions')

n_tn = 0
n_fn = 0
n_fp = 0
n_tp = 0
print('writing predictions')
for i in range(len(prediction)):
    outfile.write(str(prediction[i])+";"+str(y_test[i])+'\n')
    if prediction[i][0] > 0.5 and y_test[i][0] > 0.5:
        prefix = 'TN_'
        if predictions_to_volume:
            vis_pred.updateSegmentData(pos_test[i],1)
        n_tn += 1
    elif prediction[i][0] > 0.5 and y_test[i][1] > 0.5:
        prefix = 'FN_'
        if predictions_to_volume:
            vis_pred.updateSegmentData(pos_test[i],2)
        n_fn += 1
    elif prediction[i][1] > 0.5 and y_test[i][0] > 0.5:
        prefix = 'FP_'
        if predictions_to_volume:
            vis_pred.updateSegmentData(pos_test[i],3)
        n_fp += 1
    elif prediction[i][1] > 0.5 and y_test[i][1] > 0.5:
        prefix = 'TP_'
        if predictions_to_volume:
            vis_pred.updateSegmentData(pos_test[i],4)
        n_tp += 1
outfile.close()
if predictions_to_volume:
    vis_pred.writeSegments()

print('Number of true negatives = ',n_tn)
print('Number of false negatives = ',n_fn)
print('Number of false positives = ',n_fp)
print('Number of true positives = ',n_tp)
print('Accuracy = ',(n_tp+n_tn)/len(prediction))
print('Precision = ',n_tp/(n_tp+n_fp))

print('Metrics = ', model.metrics_names)

print(model.evaluate(x_test,y_test))
#print(model.test_on_batch(x_test,y_test))

