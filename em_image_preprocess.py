# Copyright (c) 2018, Science and Technology Facilities Council
# This software is distributed under a BSD licence. See LICENSE.txt.

# This file started with Liza's sobel_edge.py and canny_edge.py 

# Python 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import imageio
import numpy as np
from scipy import ndimage as ndi
import skimage.io as save
from skimage.filters import sobel
from skimage import img_as_uint, feature

import warnings
warnings.filterwarnings("ignore")

# options 'sobel', 'canny'
output_mode = 'sobel'

dirin = 'EM_slices_dataset_blurred_1axis'
dirout = 'EM_slices_blurred_1axis_sobel'

if not os.path.isdir(dirout):
  print("Creating output directory ...")
  os.mkdir(dirout)

listing = os.listdir(dirin)

# determine input image sizes, assuming all the same
im_first = imageio.imread(dirin + '/' + listing[0])
nx = im_first.shape[0]
ny = im_first.shape[1]
nch = im_first.shape[2]
print("Input images are ",nx, " x ", ny, ", with ",nch, " channels.")

for i in listing:
   im = imageio.imread(dirin + '/' + i)
   # use first channel only
   image1 = im[:,:,0]
   if output_mode == 'sobel': 
        edge_sobel = sobel(image1)
        edge_sobel = np.broadcast_to(edge_sobel[..., np.newaxis], (nx, ny, nch))
        save.imsave(dirout+'/'+i, edge_sobel)
   elif output_mode == 'canny':
        edges = feature.canny(image1, sigma=3)
        edges = np.broadcast_to(edges[..., np.newaxis], (nx, ny, nch))
        save.imsave(dirout+'/'+i, img_as_uint(edges))
