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

# directory with input slices
dirin = 'EM_slices_dataset_blurred_1axis'
# index file for output slices
index_file = 'EM_slices_dataset_blurred_1axis.idx'
# output predictions file
predout = 'predictions.txt'
# file for output weights
weights_out = 'model.hdf5'

# model to use
model_choice = 'custom1'

# load weights from previous run?
load_weights = False
weights_file = 'my_results_keep/model_custom1_18750raw.hdf5'
# use data augmentation
data_augment = False

### dimensions (from test image?)
rows = 48
cols = 48
ch = 3

# parameters for training
batch_size = 32
epochs = 5
learning_rate = 0.001

def inputTrainingImages(dirin,input_shape):

    ### 1st example
    # directory with EM screenshots
    # these are 1307 x 919, 8-bit/color RGBA
    # size is 1307 x 919 x 4 channels
    ##listing = os.listdir('image_dataset')
    indexfile = open(index_file,'r').read().split("\n")
    indexfile.pop()
    listing = []
    for i in indexfile:
        i = i.split(" ")
        listing.append(i)
    num_samples = len(listing)
    bad_images = listing[:]
    good_images = []
    counter = 0
    for image in listing:
        if image[0].startswith("g"):
            good_images.append(image)
            bad_images.remove(image)
            counter+=1

    newarray = np.random.choice(range(0,len(bad_images)),counter)
    bad_random = []
    for i in newarray:
        bad_random.append(bad_images[i])
    newlisting = good_images + bad_random
    newindex = open("EM_slices_dataset_blurred_1axis_new.idx","w+")
    for i in newlisting:
        newindex.write(i[0]+" "+i[1]+" "+i[2]+" "+i[3]+" "+i[4]+" "+i[5]+" "+i[6]+" "+i[7]+"\n")
    newindex.close()

    ### 2nd example
    index_file2= "EM_slices_dataset_blurred_1axis_new.idx"
    indexfile = open(index_file2,'r')
    print('number of training images = ',num_samples)

    # read in images, and flatten each image
    # haven't set dtype
    immatrix = zeros((num_samples,rows*cols*ch))
    label = zeros(num_samples)
    source_pos = zeros((num_samples,6),dtype=int)
    i = 0
    n_good = 0
    n_bad = 0

    line = indexfile.readline()
    while line != "":
        words = line.split()
        im = words[0]
        # matplotlib converts RGB to 3*floats
        immatrix[i] = array(plt.imread(dirin + '/' + im)).flatten()
        if im[0] == 'g':
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
    print("Images read in, of which ",n_good," labelled good and ",n_bad, "labelled bad.")

    # from sklearn, shuffles training data
    immatrix,label,source_pos = shuffle(immatrix, label, source_pos, random_state=2)
    print('x_train shape after shuffle: ', immatrix.shape)
    print('y_train shape after shuffle: ', label.shape)

    # reshape, the images were flattened above
    n_train = int(0.8*num_samples)
    print('Using ',n_train,' images for training')
    x_train = immatrix[:n_train].reshape(n_train,input_shape[0],input_shape[1],input_shape[2])
    y_train = to_categorical(label[:n_train])
    #x_train = x_train.astype('float32')
    #x_train /= 255
    print('x_train shape after reshape: ', x_train.shape) 
    print('y_train shape after reshape: ', y_train.shape)

    n_test = num_samples - n_train
    print('Using ',n_test,' images for testing')
    x_test = immatrix[n_train:].reshape(n_test,input_shape[0],input_shape[1],input_shape[2])
    y_test = to_categorical(label[n_train:])
    source_pos_test = source_pos[n_train:]
    print('x_test shape after reshape: ', x_test.shape)
    print('y_test shape after reshape: ', y_test.shape)
    print('source_pos_test shape after reshape: ', source_pos_test.shape)

    return x_train, y_train, x_test, y_test, source_pos_test

class mapModel(Sequential):
    """
       Our multi-layer model for maps
    """

    def __init__(self):
        Sequential.__init__(self)
 
    def createCustom1(self,input_shape):
        ## Block 1 from VGG16
        # first layer has to specify input shape but without (first) 
        # batch dimension
        self.add(Conv2D(32, (3, 3), kernel_initializer=initializers.he_normal(), activation='relu', padding='same', name='block1_conv1', input_shape=input_shape))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))

        ## include_top block from VGG16
        # flatten to 1D array
        self.add(Flatten(name='flatten'))
        self.add(Dense(80, kernel_initializer=initializers.he_normal(), activation='relu', name='fc1'))
        #self.add(Dropout(0.5))
        self.add(Dense(2, activation='softmax', name='predictions'))

    def loadVGG16(self,input_shape):
        '''16-layer network used by the VGG team in the ILSVRC-2014 
        competition. Assumes 3-channel 224x224 images.'''
        vgg_model = vgg16.VGG16(weights=None,
                                input_shape=input_shape,
                                    include_top=True,
                                    classes=2)
        for vgg_layer in vgg_model.layers:
            self.add(vgg_layer)

    def loadInception_v3(self,input_shape):
        '''Looks complicated! Assumes 3-channel 299x299 images.'''
        Iv3_model = inception_v3.InceptionV3(weights=None)
        self.add(Iv3_model)

    def loadResnet50(self,input_shape):
        '''Looks complicated! Assumes 3-channel 224x224 images.'''
        resnet50_model = resnet50.ResNet50(weights=None)
        self.add(resnet50_model)

    def loadMobileNet(self,input_shape):
        '''MobileNet is a general architecture and can be used for 
        multiple use cases.
        Requires TensorFlow backend.'''
        mobilenet_model = mobilenet.MobileNet(weights=None,
                                              input_shape=input_shape,
                                              include_top=False,
                                              pooling='max')
        self.add(mobilenet_model)

class mapVolumes():
    """ 
    3d volumes containing input maps and output predictions
    """
    def initialiseSegments(self):
        self.seg = h5py.File("predictions.seg", "w")
        self.seg.attrs.create("format","segger")
        self.seg.attrs.create("format_version",2)
        self.seg.attrs.create("name","predictions.seg")
        self.seg.attrs.create("map_size",(234, 234, 234))
        self.seg.attrs.create("map_level",0.0)
        self.seg.attrs.create("map_path","C:\Users\mdw45\VirtualBox share\emd_8194.map")
        self.seg.create_dataset("region_ids", data=(1,2,3,4))
        self.seg.create_dataset("region_colors", 
                                  data=((0.517538, 0.502082, 0.749469, 1),
                                        (0.760794, 0.871215, 0.772255, 1),
                                        (0.730948, 0.917238, 0.891777, 1),
                                        (0.952511, 0.797501, 0.998703, 1)))
        self.seg.create_dataset("ref_points", 
                                  data=((1, 1, 1),
                                        (1, 1, 1),
                                        (1, 1, 1),
                                        (1, 1, 1)))
        self.seg.create_dataset("parent_ids", 
                                  data=(0,0,0,0))
        self.dset = self.seg.create_dataset("mask", (234, 234, 234), dtype='uint32')

    def updateSegmentData(self,pos,segid):
        # empirically, we need reverse order ... WHY?
        self.dset[pos[4]:pos[5],pos[2]:pos[3],pos[0]:pos[1]] = segid

    def writeSegments(self):
        self.seg.close()
        

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

## shape set by "data_format", default is "channels_last"
x_train,y_train,x_test,y_test,pos_test = inputTrainingImages(dirin,(rows,cols,ch))

### Define the NN model, based on shape of images
model = mapModel()
if model_choice == 'custom1':
    model.createCustom1((rows,cols,ch))
elif model_choice == 'VGG16':
    model.loadVGG16((rows,cols,ch))
else:
    print("Unrecognised model, stopping ...")
    exit(1)

#plot_model(model, to_file='model.png')
print(model.layers[1].data_format)
for layer in model.layers:
    print("Layer ...")
    print("  ",layer.name)
    print("  ",layer.input_shape)
    print("  ",layer.output_shape)

# initiate RMSprop optimizer
opt = keras.optimizers.SGD(lr=learning_rate)

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
    if data_augment:
        print("Augmenting data")
        generated_data = ImageDataGenerator(featurewise_center=True, 
                                            featurewise_std_normalization=True, 
                                            samplewise_center=False, 
                                            samplewise_std_normalization=False, 
                                            zca_whitening=False, 
                                            rotation_range=0,  
                                            width_shift_range=0.1, height_shift_range=0.1, 
                                            horizontal_flip = True, vertical_flip = False)
        generated_data.fit(x_train)
        model.fit_generator(generated_data.flow(x_train, y_train, batch_size=batch_size), 
                            epochs=epochs, 
                            callbacks=[])
    else:
        model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks = [])
    print('Model trained')
    model.save_weights(weights_out)

### Now test
prediction = model.predict(x_test,batch_size=batch_size)

vis_pred = mapVolumes()
vis_pred.initialiseSegments()

fileout = predout
outfile = open(fileout,'w')
n_tn = 0
n_fn = 0
n_fp = 0
n_tp = 0
print('writing predictions')
for i in range(len(prediction)):
    outfile.write(str(prediction[i])+";"+str(y_test[i])+'\n')
    if prediction[i][0] > 0.5 and y_test[i][0] > 0.5:
        prefix = 'TN_'
        vis_pred.updateSegmentData(pos_test[i],1)
        n_tn += 1
    elif prediction[i][0] > 0.5 and y_test[i][1] > 0.5:
        prefix = 'FN_'
        vis_pred.updateSegmentData(pos_test[i],2)
        n_fn += 1
    elif prediction[i][1] > 0.5 and y_test[i][0] > 0.5:
        prefix = 'FP_'
        vis_pred.updateSegmentData(pos_test[i],3)
        n_fp += 1
    elif prediction[i][1] > 0.5 and y_test[i][1] > 0.5:
        prefix = 'TP_'
        vis_pred.updateSegmentData(pos_test[i],4)
        n_tp += 1
    if i < 100:
        ### pillow
        imgout_filename = 'raw_predictions/'+prefix+'2984_bl_'+str(i)+'.png'
        xout = (x_test[i]*255).astype(uint8)
        img_out = Image.fromarray(xout, 'RGB')
        img_out.save(imgout_filename)
outfile.close()
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

exit(1)

# to view intermediate layers, set up partial model convering
# required steps
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('conv2d_1').output)
intermediate_output = intermediate_layer_model.predict(x_train)

plt.subplot(1, 2, 1)
plt.imshow(x_train[0].reshape(rows,cols,ch))
plt.subplot(1, 2, 2)
plt.imshow(intermediate_output[0])
##plt.show()
