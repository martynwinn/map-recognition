import os
import imageio
import skimage.io as save
from skimage import img_as_uint
import numpy as np
from scipy import ndimage as ndi
from skimage import feature

import warnings
warnings.filterwarnings("ignore")

dirin = 'EM_slices_dataset_blurred_1axis'
dirout = 'EM_slices_noisyedge'

listing = os.listdir(dirin)
for i in listing:
	im = imageio.imread(dirin + '/' + i)
	image1 = im[:,:,0]
	edges = feature.canny(image1, sigma=3)
	edges = np.broadcast_to(edges[..., np.newaxis], (48, 48, 3))
	save.imsave(dirout+'/'+i, img_as_uint(edges))

