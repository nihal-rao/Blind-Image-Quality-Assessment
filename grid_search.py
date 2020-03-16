import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models
import transforms 
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import os
import copy
from PIL import Image
import json
from load_data import build_dataset

def emd_loss(target, output):
	p_sum = torch.cumsum(output,dim=1)
	q_sum = torch.cumsum(target,dim=1)
	loss = torch.mean(torch.sqrt(torch.mean((q_sum-p_sum)**2,1)))
	return loss

if __name__ == '__main__':
	pth_path = r'/home/mancmanomyst/tid/saved_models/'
	path_to_json_train = r'/home/mancmanomyst/tid/tid_labels_train.json'
	path_to_json_test = r'/home/mancmanomyst/tid/tid_labels_test.json'
	path_to_imgs = r'/home/mancmanomyst/tid/distorted_images/'
	TID_train,TID_val,TID_test = build_dataset(path_to_imgs,path_to_json_train,path_to_json_test)
	
	data_loader_train = DataLoader(TID_train,batch_size=16,num_workers=2,shuffle=True)
	data_loader_val = DataLoader(TID_val,batch_size=16,num_workers=2)

	device = torch.device('cuda')
	
	
	
	n_search = 20
	cache = {}

	for n in range(n_search):


		model = models.mobilenet_v2(pretrained=True)
		model.classifier = nn.Sequential(nn.Dropout(0.75),
			nn.Linear(1280, 10),nn.Softmax())
		model = model.to(device)
		conv_base_lr = 10**(np.random.uniform(low =-6,high=-3))
		dense_lr = 10**(np.random.uniform(low=-6,high=-3))
		optimizer = optim.Adam([
		{'params': model.features.parameters(), 'lr': conv_base_lr},
		{'params': model.classifier.parameters(), 'lr': dense_lr}],
		)
		num_epochs = 10
		val_losses = []

		for epoch in range(num_epochs):
			model.train()
			for i,batch in enumerate(data_loader_train):
				imgs = batch['img'].to(device)
				labels = batch['label'].to(device)
				outputs = model(imgs)
				outputs = outputs.view(-1,10)

				optimizer.zero_grad()

				loss = emd_loss(labels, outputs)

				loss.backward()

				optimizer.step()

			#print('Epoch: {}/{} | Step: {}/{} | Training EMD loss: {0:.4f}'.format(epoch + 1, num_epochs, i + 1, len(TID_train) // 32 + 1, loss.data[0]))

			batch_val_losses = []
			model.eval()
		
			for i,batch in enumerate(data_loader_val):
				images = batch['img'].to(device)
				labels = batch['label'].to(device)
				with torch.no_grad():
					outputs = model(images)
				outputs = outputs.view(-1, 10)
				val_loss = emd_loss(labels, outputs)
				batch_val_losses.append((val_loss.item())*(images.size()[0]))
			avg_val_loss = sum(batch_val_losses)/len(TID_val)
			val_losses.append(avg_val_loss)
		cache[n] = [sum(val_losses)/len(val_losses),conv_base_lr,dense_lr]
		print('conv_base_lr : {} dense_lr : {} val_loss : {:.4f}'.format(conv_base_lr,dense_lr, sum(val_losses)/len(val_losses)))

