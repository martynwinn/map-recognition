# Copyright (c) 2018, Science and Technology Facilities Council
# This software is distributed under a BSD licence. See LICENSE.txt.

# Python 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as pimg

# This is the CCP-EM library for i/o to MRC format map files
# https://pypi.org/project/mrcfile/
import mrcfile

### high level options
# second reference for multi-class problems
second_ref = True
# output histograms for each sub-image
output_histogram = True
# output individual images for classification
output_images = False

### files
# input map from which to extract 2D slices
input_map_file = 'emd_7024.map'
# reference map from fitted PDB for annotation
reference_map_file = 'emd_7024_from_pdb_prot.mrc'
# second reference map from fitted PDB for annotation
reference2_map_file = 'emd_7024_from_pdb_na.mrc'
# output directory for slices
dirout = 'emd_7024_slices_1axis_histo'
# index file for output slices
index_file = 'emd_7024_slices_1axis_histo.idx'

### options
#np.set_printoptions(threshold=np.inf)
# list of axes to section along, typically ['Z'] for 1axis or ['X','Y','Z'] for 3axes
axes_list = ['Z']
# linear dimension of slice
window_size = 48
# offset from one slice to the next within a section
offset = 10
# offset from one section to the next
section_skip = 10
# normalise values in a slice to be between map_min and map_max
normalise = True
minmax_auto = True
# if minmax_auto then following manual values over-written
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
if minmax_auto:
  map_min = np.amin(mrc.data)
  map_max = np.amax(mrc.data)

print("Opening reference map file ...")
mrcref = mrcfile.open(reference_map_file)
mrcref.print_header()

if second_ref:
  print("Opening second reference map file ...")
  mrcref2 = mrcfile.open(reference2_map_file)
  mrcref2.print_header()

if not os.path.isdir(dirout):
  print("Creating output directory ...")
  os.mkdir(dirout)

indexfile = open(index_file,'w')

# Loop over map data block 3 times, each time sectioning along different axis.
# The map data block is copied to a temporary array loc_mrcdata, and we use
# np.transpose to change the axis order, i.e. if axis is 'X' then that is
# the 3rd dimension, and slices are in the y-z plane.
for axis in axes_list:
  if axis in ['X']:
    print("transposing")
    loc_mrcdata = np.transpose(mrc.data,(1,2,0))
    loc_mrcrefdata = np.transpose(mrcref.data,(1,2,0))
    if second_ref:
      loc_mrcref2data = np.transpose(mrcref2.data,(1,2,0))
  elif axis in ['Y']:
    print("transposing")
    loc_mrcdata = np.transpose(mrc.data,(2,0,1))
    loc_mrcrefdata = np.transpose(mrcref.data,(2,0,1))
    if second_ref:
      loc_mrcref2data = np.transpose(mrcref2.data,(2,0,1))
  else:
    loc_mrcdata = mrc.data
    loc_mrcrefdata = mrcref.data
    if second_ref:
      loc_mrcref2data = mrcref2.data

  # parameters for grid plots
  grid_rows = int((loc_mrcdata.shape[0]-window_size)/offset) + 1
  grid_cols = int((loc_mrcdata.shape[1]-window_size)/offset) + 1
  figsize_x = grid_cols*2.0
  figsize_y = grid_rows*2.0
  
  for section in range(0,loc_mrcdata.shape[2],section_skip):

    print('processing section ',section)
    igrid = 0
    jgrid = 0
    # images grid
    fig1, axs1 = plt.subplots(grid_rows,grid_cols,num=1,sharex=True,sharey=True,figsize=[figsize_x,figsize_y])
    # histogram grid
    if output_histogram:
      fig2, axs2 = plt.subplots(grid_rows,grid_cols,num=2,sharex=True,sharey=True,figsize=[figsize_x,figsize_y])

    for col in range(0,loc_mrcdata.shape[1]-window_size,offset):

        for row in range(0,loc_mrcdata.shape[0]-window_size,offset):

            myslice = loc_mrcdata[row:row+window_size,col:col+window_size,section].copy(order='C')

            # annotate
            annotate = 'b_'
            annotate_colour = 'black'
            n_ref_dens = 0
            for x in range(row,row+window_size):
                for y in range(col,col+window_size):
                    #print(x,y,loc_mrcrefdata[x,y,section])
                    if loc_mrcrefdata[x,y,section] > ref_thresh:
                        n_ref_dens += 1
            # if a fraction of grid points in reference map have significant
            # density, then label as good
            if n_ref_dens > window_size*window_size*good_fraction:
                annotate = 'p_'
                annotate_colour = 'blue'
            if second_ref:
              n_ref2_dens = 0
              for x in range(row,row+window_size):
                for y in range(col,col+window_size):
                    #print(x,y,loc_mrcref2data[x,y,section])
                    if loc_mrcref2data[x,y,section] > ref_thresh:
                        n_ref2_dens += 1
              # if a fraction of grid points in reference map have significant
              # density, then label as good
              if n_ref2_dens > window_size*window_size*good_fraction and n_ref2_dens > n_ref_dens:
                annotate = 'n_'
                annotate_colour = 'orange'

            # So far, myslice has the original map values. By default, the output images
            # are 8-bit, so we need to put these into the range 0 - 255. We use np.clip
            # so that a given range of map values map into 0 - 255. Lower values are set
            # to 0 and higher values set to 255. We can vary this, to enhance the features
            # in the image (might also be done with data augmentation routines from Keras).
            if normalise:
                myslice = 255.0 * np.clip((myslice - map_min) / (map_max - map_min),0,1)

            # construct histogram
            #### np for binning?
            #### need normalised values?
            if output_histogram:
              if normalise:
                hist_min = 0.0
                hist_max = 255.0
              else:
                hist_min = map_min
                hist_max = map_max
              (histo,edges) = np.histogram(myslice, bins=10, range=(hist_min,hist_max))

            ### pillow
            # output the 2D slices as standard PNG files, for input to the machine learning
            imgout_filename = annotate + '2984_bl_'+str(section)+'_'+str(col)+'_'+str(row)+axis+'.png'
            img_out = pimg.fromarray(myslice)
            # Write as RGB ... it is really greyscale, so red=green=blue at every pixel.
            # This must be wasteful, but the ML I started with assumed RGB and it was
            # easiest to go with it.
            img_new = img_out.convert('RGB')
            if output_images:
              img_new.save(dirout + '/' + imgout_filename)

            axs1[igrid][jgrid].imshow(img_new)
            for spine in axs1[igrid][jgrid].spines.values():
              spine.set_edgecolor(annotate_colour)
              spine.set_linewidth(4.0)
            if output_histogram:
              axs2[igrid][jgrid].bar(edges[:-1],histo,width=20,align='edge')
              for spine in axs2[igrid][jgrid].spines.values():
                spine.set_edgecolor(annotate_colour)
                spine.set_linewidth(4.0)
            jgrid += 1
            if jgrid == grid_cols:
              jgrid = 0
              igrid += 1

            line = imgout_filename+' '+axis+' '+str(row)+' '+str(row+window_size)+' '+str(col)+' '+str(col+window_size)+' '+str(section)+' '+str(section+1)
            if output_histogram:
              line = line + ' ' + ' '.join(str(h) for h in histo)
            indexfile.write(line+'\n')

    plt.figure(num=1)
    img_plot_title = dirout + '/' + 'imgs_'+str(section)+'.png'
    plt.savefig(img_plot_title)
    plt.close()
    if output_histogram:
      plt.figure(num=2)
      histo_plot_title = dirout + '/' + 'hists_'+str(section)+'.png'
      plt.savefig(histo_plot_title)
      plt.close()

indexfile.close()
