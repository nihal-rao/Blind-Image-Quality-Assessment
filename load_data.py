import torch
import numpy as np
import transforms 
from torch.utils.data import Dataset, DataLoader, ConcatDataset,Subset
import os
from PIL import Image
import json
import random

class TIDLoader(Dataset):
	def __init__(self,path_to_imgs,path_to_json,transform): 
		self.path_to_json = path_to_json
		self.path_to_imgs = path_to_imgs
		self.transform = transform
		f = json.load(open(path_to_json))
		self.image_ids = [i['image_id'] for i in f]
		self.labels = [i['label'] for i in f]
	def __getitem__(self,idx):
		img_id = self.image_ids[idx]
		img = Image.open(self.path_to_imgs + img_id + '.bmp')
		label = self.labels[idx]
		label = torch.tensor(label)
		img, label = self.transform(img, label)
		data = {}
		data['img']=img
		data['label']=label
		return data
	def __len__(self):
		return len(self.image_ids)


def build_dataset(path_to_imgs,path_to_json_train,path_to_json_test):	
	data_transforms = {
	'train': transforms.Compose([
		transforms.Resize(256),
		transforms.CustomCrop(),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])}
	TID_train = TIDLoader(path_to_imgs,path_to_json_train,data_transforms['train'])
	TID_test = TIDLoader(path_to_imgs,path_to_json_test,data_transforms['val'])
	"""
	Since ground truth distributions were in train-test split (no val data),
	train data is split in two parts.Only 76 % of original train data is used 
	for training, remaining is added to original test data. THis test data is 
	then split into validation and test data.
	In short, the total number of images available in TID2013 (3k)
	are split in a ratio of 70-20-10 (train-val-test). 
	"""
	train_len=len(TID_train)
	idx = list(range(train_len))
	random.shuffle(idx)
	split_idx = idx[:int(0.76*train_len)]
	train_split = Subset(TID_train,split_idx)
	
	split_idx = idx[int(0.76*train_len):]
	train_val_split = Subset(TID_train,split_idx)
	val_split = ConcatDataset([train_val_split,TID_test])
	
	val_len = len(val_split)
	val_idx = list(range(val_len))
	random.shuffle(val_idx)
	val_split_idx = val_idx[:int(0.75*val_len)]
	final_val_split = Subset(val_split,val_split_idx)
	
	test_split_idx = val_idx[int(0.75*val_len):]
	test_split = Subset(val_split,test_split_idx)
	return train_split,final_val_split,test_split

if __name__ == '__main__':
	path_to_json_train = r'/home/mancmanomyst/tid/tid_labels_train.json'
	path_to_json_test = r'/home/mancmanomyst/tid/tid_labels_test.json'
	path_to_imgs = r'/home/mancmanomyst/tid/distorted_images/'
	TID_train,TID_val,TID_test = build_dataset(path_to_imgs,path_to_json_train,path_to_json_test)
	print(len(TID_train))
	print(len(TID_val))
	print(len(TID_test))

	"""
	for i in range(len(TID_train)):
		train_sample = TID_train[i]
		n = train_sample['img'].size()[2]
		if n !=224:
			print(i)
			break
	print('NOW TEST')
	for i in range(len(TID_test)):
		test_sample = TID_test[i]
		n = test_sample['img'].size()[2]
		if n !=224:
			print(i)
			break

	#data_loader_val = DataLoader(TID_val,batch_size=16,num_workers=2)
	
	
	for batch in data_loader_val:
		images = batch['img']
		labels = batch['label']
	"""
