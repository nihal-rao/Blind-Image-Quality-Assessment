# Image-Quality-Assessment
An implementation of the NIMA paper on the TID2013 dataset, using PyTorch.

## TID2013 
Introduced in [this paper](https://www.sciencedirect.com/science/article/pii/S0923596514001490),TID2013 contains 3000 images obtained from 25 reference images with 24 types of distortions for each reference image, and 5 levels for each type of distortion.
Each image is labelled with a Mean Opinion Score (MOS) ranging from 0-9.
Some distortions included are :
* Additive and Multiplicative Gaussian Noise
* JPEG Compression
* Chromatic Aberration
* Contrast change
