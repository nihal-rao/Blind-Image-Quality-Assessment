import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models
import transforms 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from load_data import build_dataset
import math
import scipy.stats

def emd_loss(target, output):
	p_sum = torch.cumsum(output,dim=1)
	q_sum = torch.cumsum(target,dim=1)
	loss = torch.mean(torch.sqrt(torch.mean((q_sum-p_sum)**2,1)))
	return loss
	
def srcc_conf_bounds(r,num):
	stderr = 1.0 / math.sqrt(num - 3)
	delta = 1.96 * stderr
	lower = math.tanh(math.atanh(r) - delta)
	upper = math.tanh(math.atanh(r) + delta)
	return upper,lower

if __name__ == '__main__':
	pth_path = r'/home/mancmanomyst/tid/saved_models/'
	path_to_json_train = r'/home/mancmanomyst/tid/tid_labels_train.json'
	path_to_json_test = r'/home/mancmanomyst/tid/tid_labels_test.json'
	path_to_imgs = r'/home/mancmanomyst/tid/distorted_images/'
	TID_train,TID_val,TID_test = build_dataset(path_to_imgs,path_to_json_train,path_to_json_test)
	
	data_loader_train = DataLoader(TID_train,batch_size=16,num_workers=2,shuffle=True)
	data_loader_val = DataLoader(TID_val,batch_size=16,num_workers=2)

	device = torch.device('cuda')
	
	model = models.mobilenet_v2(pretrained=True)
	model.classifier = nn.Sequential(nn.Dropout(0.75),
			nn.Linear(1280, 10),nn.Softmax())
	model = model.to(device)
	
	conv_base_lr = 0.00015385448485299424
	dense_lr = 3.752717909839667e-05
	optimizer = optim.Adam([
		{'params': model.features.parameters(), 'lr': conv_base_lr},
		{'params': model.classifier.parameters(), 'lr': dense_lr}],
		)
	num_epochs = 100
	best_epoch = 0
	count = 0
	init_val_loss = float('inf')
	train_losses = []
	val_losses = []

	for epoch in range(num_epochs):
		batch_loss=[]
		model.train()
		i=0
		for i,batch in enumerate(data_loader_train):
			imgs = batch['img'].to(device)
			labels = batch['label'].to(device)
			outputs = model(imgs)
			outputs = outputs.view(-1,10)

			optimizer.zero_grad()

			loss = emd_loss(labels, outputs)
			batch_loss.append((loss.item())*(imgs.size()[0]))

			loss.backward()

			optimizer.step()

			#print('Epoch: {}/{} | Step: {}/{} | Training EMD loss: {0:.4f}'.format(epoch + 1, num_epochs, i + 1, len(TID_train) // 32 + 1, loss.data[0]))
		
		avg_loss = sum(batch_loss)/len(TID_train)
		train_losses.append(avg_loss)
		print('Epoch {} averaged training EMD loss: {:.4f}'.format(epoch + 1, avg_loss))
		
		"""
		if (epoch + 1) % 10 == 0:
				conv_base_lr = conv_base_lr * 0.95
				dense_lr = dense_lr * 0.95
				optimizer = optim.Adam([
					{'params': model.features.parameters(), 'lr': conv_base_lr},
					{'params': model.classifier.parameters(), 'lr': dense_lr}],
				)
		"""
		
		batch_val_losses = []
		model.eval()
		i=0
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

			# lrs.send('val_emd_loss', avg_val_loss)

		print('Epoch {} completed. Averaged EMD loss on val set: {:.4f}'.format(epoch + 1, avg_val_loss))

			# Use early stopping to monitor training
		if avg_val_loss < init_val_loss:
			init_val_loss = avg_val_loss
			torch.save(model.state_dict(), os.path.join(pth_path, 'epoch'+str(epoch)+'.pth'))
			count = 0
			best_epoch = epoch
		elif avg_val_loss >= init_val_loss:
			count += 1
			if count == 5:
				print('Val EMD loss has not decreased in {} epochs. Training terminated.'.format(5))
				break

	print('Training completed.')

	epochs = range(1, epoch + 2)
	plt.plot(epochs, train_losses, 'b-', label='train loss')
	plt.plot(epochs, val_losses, 'g-', label='val loss')
	plt.title('EMD loss')
	plt.legend()
	plt.savefig('./loss.png')

	best_path = 'epoch'+str(best_epoch)+'.pth'
	best_path = os.path.join(pth_path,best_path)
	model.load_state_dict(torch.load(best_path))

	data_loader_test = DataLoader(TID_test,batch_size=16,num_workers=2)
	batch_test_losses = []
	model.eval()
	i=0
	mos_idx = torch.arange(10)
	mos_op_list = []
	mos_gt_list = []
	mos_idx = mos_idx.to(device)
	for i,data in enumerate(data_loader_test):
		images = data['img'].to(device)
		labels = data['label'].to(device)
		with torch.no_grad():
			outputs = model(images)
		outputs = outputs.view(-1, 10)
		test_loss = emd_loss(labels, outputs)
		mos_op = torch.mean(outputs*mos_idx,1)
		mos_gt = torch.mean(labels*mos_idx,1)
		mos_op_list.extend(mos_op.cpu().detach().numpy())
		mos_gt_list.extend(mos_gt.cpu().detach().numpy())
		batch_test_losses.append((test_loss.item())*(images.size()[0]))
	avg_test_loss = sum(batch_test_losses)/len(TID_test)
	print('Test EMD loss is {:.4f}'.format(avg_test_loss))
	mos_op_list = np.asarray(mos_op_list)
	mos_gt_list = np.asarray(mos_gt_list)
	r,_= scipy.stats.spearmanr(mos_op_list, mos_gt_list, axis=0)
	num = mos_gt_list.shape[0]
	upper,lower = srcc_conf_bounds(r,num)
	print(r)
	print("SRCC  95 % confidence interval, lower {} upper {}".format(lower, upper))