
# Python 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys

import numpy as np
from PIL import Image as pimg

# This is the CCP-EM library for i/o to MRC format map files
# https://pypi.org/project/mrcfile/
import mrcfile

### files
# input map from which to extract 2D slices
input_map_file = 'emd_2984_sdev_1_0.mrc'
# reference map from fitted PDB for annotation
reference_map_file = 'emd_2984_from_pdb.mrc'
# output directory for slices
dirout = 'EM_slices_dataset_blurred_1axis'

### options
#np.set_printoptions(threshold=np.inf)
# list of axes to section along
axes_list = ['Z']
# linear dimension of slice
window_size = 48
# offset from one slice to the next within a section
offset = 10
# offset from one section to the next
section_skip = 10
# normalise values in a slice to be between map_min and map_max
normalise = True
#map_min = np.amin(mrc.data)
#map_max = np.amax(mrc.data)
map_min = -0.02
map_max = 0.05
# threshold for counting ref map density
ref_thresh = 0.1
# if a fraction of grid points in reference map have significant
# density, then label as good
good_fraction = 0.05

### don't edit below here

print("Opening input map file ...")
mrc = mrcfile.open(input_map_file)
mrc.print_header()

print("Opening reference map file ...")
mrcref = mrcfile.open(reference_map_file)
mrcref.print_header()

if not os.path.isdir(dirout):
  print("Creating output directory ...")
  os.mkdir(dirout)

# loop over map data block 3 times, each time sectioning along different axis
for axis in axes_list:
  if axis in ['X']:
    print("transposing")
    loc_mrcdata = np.transpose(mrc.data,(1,2,0))
    loc_mrcrefdata = np.transpose(mrcref.data,(1,2,0))
  elif axis in ['Y']:
    print("transposing")
    loc_mrcdata = np.transpose(mrc.data,(2,0,1))
    loc_mrcrefdata = np.transpose(mrcref.data,(2,0,1))
  else:
    loc_mrcdata = mrc.data
    loc_mrcrefdata = mrcref.data
  
  for section in range(0,loc_mrcdata.shape[2],section_skip):

    print('processing section ',section)

    for col in range(0,loc_mrcdata.shape[1]-window_size,offset):

        for row in range(0,loc_mrcdata.shape[0]-window_size,offset):

            myslice = loc_mrcdata[row:row+window_size,col:col+window_size,section].copy(order='C')

            # annotate
            annotate = 'b_'
            n_ref_dens = 0
            for x in range(row,row+window_size):
                for y in range(col,col+window_size):
                    #print(x,y,loc_mrcrefdata[x,y,section])
                    if loc_mrcrefdata[x,y,section] > ref_thresh:
                        n_ref_dens += 1
            # if a fraction of grid points in reference map have significant
            # density, then label as good
            if n_ref_dens > window_size*window_size*good_fraction:
                annotate = 'g_'

            if normalise:
                myslice = 255.0 * np.clip((myslice - map_min) / (map_max - map_min),0,1)

            ### pillow
            imgout_filename = annotate + '2984_bl_'+str(section)+'_'+str(col)+'_'+str(row)+axis+'.png'
            img_out = pimg.fromarray(myslice)
            img_new = img_out.convert('RGB')
            img_new.save(dirout + '/' + imgout_filename)

