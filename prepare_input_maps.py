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
from PIL import Image as pimg

original_map = 'emd_2984.map'
fitted_coords = '5a1a.pdb'

### blur/sharpen map
# Go into Chimera
# Use Gaussian blurring in Chimera with e.g. sd 1.0
# (Volume Viewer -> Tools -> Volume Filter)
# beta-gal tested with unblurred and blurred maps, and latter slightly better
# no robust way of estimating required blurring

### create map from coords
# Go into Chimera
# open command line
# molmap #1 2.2 onGrid #0
# where #1 is the PDB file, 2.2 is the resoluton, and the grid is taken from map file #0

