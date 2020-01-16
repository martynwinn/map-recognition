# Run using anaconda environment, Python 3
# select tensorflow backend for keras
#export LD_LIBRARY_PATH=/home/mdw/anaconda3/lib
# otherwise get error
#ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.9' not found

import numpy as np
import time
import imp
import os

import keras
import keras.backend
import keras.models

import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.tests.networks.base
import innvestigate.utils.visualizations as ivis

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
        self.add(Dense(1024, kernel_initializer=initializers.he_normal(), activation='relu', name='fc1'))
        #self.add(Dropout(0.5))
        self.add(Dense(2, activation='softmax', name='predictions'))

model = mapModel()
model.createCustom1((48,48,c3))
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
print('Model compiled')
model.load_weights("my_results_keep/model_custom1_33148.hdf5")
print('Weights loaded')



#with open('my_results_keep/model_custom1.yaml', 'r') as myfile:
#  model_arch = myfile.read()
#model = keras.models.model_from_yaml(model_arch)
