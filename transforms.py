import torchvision.transforms.functional as F
import torch
from PIL import Image
import numpy as np
import random

class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, image, target):
		for t in self.transforms:
			image, target = t(image, target)
		return image, target

class Resize(object):
	def __init__(self, size, interpolation=Image.BILINEAR):
		assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
		self.size = size
		self.interpolation = interpolation

	def __call__(self, img,label):
		return F.resize(img, [self.size,self.size], self.interpolation),label

class Normalize(object):
	def __init__(self, mean, std, inplace=False):
		self.mean = mean
		self.std = std
		self.inplace = inplace

	def __call__(self, img,label):
		return F.normalize(img, self.mean, self.std, self.inplace),label

class ToTensor(object):
	def __call__(self, image, target):
		image = F.to_tensor(image)
		return image, target

class CustomCrop(object):
	def __call__(self, img,label):
		crop_x = np.random.randint(low=0,high=33)
		crop_y = np.random.randint(low=0,high=33) 
		return F.crop(img,crop_y,crop_x,224,224), label

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, target):
        if random.random() < self.p:
            return F.hflip(img), target
        return img, target