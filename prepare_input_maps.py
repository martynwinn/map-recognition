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

### create map from coords
# Go into Chimera

