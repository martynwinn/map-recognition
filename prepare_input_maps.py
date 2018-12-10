# Copyright (c) 2018, Science and Technology Facilities Council
# This software is distributed under a BSD licence. See LICENSE.txt.

### Run as 
###     chimera --nogui --script prepare_input_maps.py
# tested on Chimera 1.13

# Python 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys

from chimera import runCommand as rc

### set these parameters
# browse https://www.ebi.ac.uk/pdbe/emdb/ for suitable maps
emdb_id = '6549'
# "Fitted PDB structure"
pdb_id = '3jcd'
# Nominal resolution of EMDB entry
resolution = 3.7
# We vary blurring to see how it affects recognition
sd_blurring = 1.0

## input local filenames
#original_map = 'emd_'+emdb_id+'.map'
#fitted_coords = pdb_id+'.pdb'
## or string to fetch from web
original_map = 'emdbID:'+emdb_id
fitted_coords = 'pdbID:'+pdb_id

## output filenames
# blurred map 
blurred_map = 'emd_'+emdb_id+'_blurred.mrc'
# map from all fitted coordinates
fitted_coords_map = 'emd_'+emdb_id+'_from_pdb.mrc'
# map from fitted protein coordinates
fitted_coords_map_protein = 'emd_'+emdb_id+'_from_pdb_prot.mrc'
# map from fitted nucleic acid coordinates
fitted_coords_map_na = 'emd_'+emdb_id+'_from_pdb_na.mrc'

### Chimera commands
# note that #0 etc refer to model IDs and must be kept consistent
rc("open 0 %s"%(original_map))
# chimera defaults to step 4, which confuses the later onGrid
rc("volume #0 step 1")
## NB could use "step" to get smaller map file, or use "vop resample" which interpolates
## rather than selecting data points

print("Map downloaded, now blurring")

### blur/sharpen map
# Go into Chimera
# Use Gaussian blurring in Chimera with e.g. sd 1.0
# (Volume Viewer -> Tools -> Volume Filter)
# beta-gal tested with unblurred and blurred maps, and latter slightly better
# no robust way of estimating required blurring
rc("vop gaussian #0 sd %f modelId 5"%(sd_blurring))
rc("volume #5 save %s"%(blurred_map))

rc("open 1 %s"%(fitted_coords))
# make map from model, Agnel also has sigmaFactor 0.356
rc("molmap #1 %s onGrid #0 modelId 2"%(resolution))
rc("volume #2 save %s"%(fitted_coords_map))
rc("select #1 & protein")
rc("molmap sel %s onGrid #0 modelId 3"%(resolution))
rc("volume #3 save %s"%(fitted_coords_map_protein))
rc("select #1 & nucleic acid")
rc("molmap sel %s onGrid #0 modelId 4"%(resolution))
rc("volume #4 save %s"%(fitted_coords_map_na))
rc("stop now")

