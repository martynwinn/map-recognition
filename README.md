# map-recognition

Image recognition for structural biology maps using convolutional neural networks.
This is a JRA project for [West-Life](https://west-life.eu).

The 4-stage protocol uses the following files:

* prepare_input_maps.py:  Prepare input EM map e.g. with blurring or sharpening factor to get the right degree of blobiness. Create the reference map from fitted coordinates, which provides the ground truth for training.

* extract_EM_slices.py:  This extracts a large number of 2D slices from the provided 3D volume, to be used as training / test datasets.

* em_image_preprocess.py:  The set of extracted 2D images can be preprocessed, e.g. with filters.

* em_image_classify.py:  This creates the machine-learning model, trains the network, and tests it.

Other files / directories include:

* trained_models:  Directory of selected trained models. The .yaml file specifies the architecture. The .hdf5 files have trained weights.

* examples: Currently has some Jupyter notebooks showing some uses of the code.

* plots: Example results plots, used in the wiki.

Software is released under BSD licence.

Martyn Winn
STFC


