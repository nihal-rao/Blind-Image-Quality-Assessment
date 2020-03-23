# Image-Quality-Assessment
An implementation of the NIMA paper on the TID2013 dataset, using PyTorch.

## TID2013 
Introduced in [this paper](https://www.sciencedirect.com/science/article/pii/S0923596514001490),TID2013 contains 3000 images obtained from 25 reference images with 24 types of distortions for each reference image, and 5 levels for each type of distortion.
Each image is labelled with the mean (ranging from 0-9) and standard deviation of scores.
Some distortions included are :
* Additive and Multiplicative Gaussian Noise
* JPEG Compression
* Chromatic Aberration
* Contrast change

![alt text](/images/IO3.bmp)                                                   

![alt text](/images/i03_01_5.bmp)
Reference Image                                                                   With Additive Gaussian Noise
                                                                                  MOS = 3.925,std dev. = 0.1083
