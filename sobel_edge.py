import os
import imageio
import skimage.io as save
from skimage.filters import sobel
import numpy as np

import warnings
warnings.filterwarnings("ignore")

dirin = 'EM_slices_dataset_blurred_1axis'
dirout = 'EM_slices_sobel'

listing = os.listdir(dirin)
for i in listing:
	im = imageio.imread(dirin + '/' + i)
	image1 = im[:,:,0]
	edge_sobel = sobel(image1)
	edge_sobel = np.broadcast_to(edge_sobel[..., np.newaxis], (48, 48, 3))
	save.imsave(dirout+'/'+i, edge_sobel)

