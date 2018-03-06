import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model
from keras.applications import vgg16, inception_v3, resnet50, mobilenet

import os
import numpy as np
from numpy import *
from PIL import Image
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

### dimensions (from test image?)
rows = 30
cols = 30
ch = 1

# parameters for training
batch_size = 32
epochs = 1

def inputTrainingImages(input_shape):

    ### 1st example
    # directory with EM screenshots
    # these are 1307 x 919, 8-bit/color RGBA
    # size is 1307 x 919 x 4 channels
    ##listing = os.listdir('image_dataset')

    ### 2nd example
    listing = os.listdir('EM_slices_dataset')
    num_samples = len(listing)
    print 'number of training images = ',num_samples

    # read in images, and flatten each image
    # haven't set dtype
    immatrix = zeros((num_samples,rows*cols*ch))
    label = zeros(num_samples)
    i = 0
    for im in listing:
        immatrix[i] = array(plt.imread('EM_slices_dataset/' + im)).flatten()
        if im[0] == 'g':
            label[i] = 1
        elif im[0] == 'b':
            label[i] = 0
        i += 1

    # from sklearn, shuffles training data
    immatrix,label = shuffle(immatrix, label, random_state=2)
    print 'x_train shape after shuffle: ', immatrix.shape 
    print 'y_train shape after shuffle: ', label.shape 

    # reshape, the images were flattened above
    n_train = int(0.8*num_samples)
    print 'Using ',n_train,' images for training'
    x_train = immatrix[:n_train].reshape(n_train,input_shape[0],input_shape[1],input_shape[2])
    y_train = label[:n_train]
    #x_train = x_train.astype('float32')
    #x_train /= 255
    print 'x_train shape after reshape: ', x_train.shape 
    print 'y_train shape after reshape: ', y_train.shape 

    n_test = num_samples - n_train
    print 'Using ',n_test,' images for testing'
    x_test = immatrix[:n_test].reshape(n_test,input_shape[0],input_shape[1],input_shape[2])
    y_test = label[:n_test]
    print 'x_test shape after reshape: ', x_test.shape 
    print 'y_test shape after reshape: ', y_test.shape 

    return x_train, y_train, x_test, y_test

class mapModel(Sequential):
    """
       Our multi-layer model for maps
    """

    def __init__(self):
        Sequential.__init__(self)
 
    def createCustom(self,input_shape):
        # Conv2D, 1 filter, (3,3) convolution window
        # first layer has to specify input shape but without (first) 
        # batch dimension
        self.add(Conv2D(3, (3, 3), padding='same', input_shape=input_shape))
        print self.output_shape
        self.add(MaxPooling2D(pool_size=(2, 2)))
        print self.output_shape
        # flatten to 1D array
        self.add(Flatten())
        # dense NN, output of length 128
        self.add(Dense(128, activation='relu'))
        self.add(Dropout(0.5))
        # dense NN, output of length 1 .... this needs to agree 
        # with dimension of y_train
        self.add(Dense(1, activation='softmax'))

    def loadVGG(self,input_shape):
        vgg_model = vgg16.VGG16(weights=None)
        self.add(vgg_model)

###### start here ######

## shape set by "data_format", default is "channels_last"
x_train,y_train,x_test,y_test = inputTrainingImages((rows,cols,ch))

### Define the NN model, based on shape of images
model = mapModel()
#model.createCustom((rows,cols,ch))
model.loadVGG((rows,cols,ch))

#plot_model(model, to_file='model.png')
print [layer.name for layer in model.layers]

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model 
# "loss" is used in training
# "metrics" is used to judge
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
print 'Model compiled'

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs)
print 'Model trained'

# to view intermediate layers, set up partial model convering
# required steps
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('conv2d_1').output)
intermediate_output = intermediate_layer_model.predict(x_train)

plt.subplot(1, 2, 1)
plt.imshow(x_train[0].reshape(30,30))
plt.subplot(1, 2, 2)
plt.imshow(intermediate_output[0])
##plt.show()

### Now test
prediction = model.predict(x_test)
print prediction.shape

print 'Metrics = ', model.metrics_names

print model.evaluate(x_test,y_test)
print model.test_on_batch(x_test,y_test)
