import torch
import numpy as np


def train_model(model, dataset, epochs, batch_size, lr, cuda):
	'''
	Function for training a model. 
	'''

	model.train()
	optim = torch.optim.Adam(model.parameters(), lr=lr)
	total_loss_list = []
	ce_loss_list = []
	KLD_loss_list = []
	print('Starting to train!')

	for epoch in range(1, epochs+1):
		batch_idx = np.random.permutation(dataset.shape[0])
		for i in range((len(batch_idx)//batch_size)+1):
			if cuda:
				batch = torch.as_tensor(dataset[batch_idx[i*batch_size:(i+1)*batch_size]]).cuda()
			else:
				batch = torch.from_numpy(dataset[batch_idx[i*batch_size:(i+1)*batch_size]]).float()
			print(batch.shape)
			optim.zero_grad()
			reconstructed_x, mu, logvar = model.forward(batch)

			##Calculating Reconstruction loss and KLD loss, from method in model Class.
			total_loss, ce_loss, KLD_loss = model.calc_loss(reconstructed_x, batch, mu, logvar)
			total_loss.backward()
			optim.step()
			
			##Append loss to list to visualize model loss improvement.
			total_loss_list.append(total_loss.item())
			ce_loss_list.append(ce_loss.item())
			KLD_loss_list.append(KLD_loss.item())
			print('Total loss: {}, ce_loss: {}, KLD_loss: {}'.format(total_loss.item(), ce_loss.item(), KLD_loss.item() ))
			if i == 2:
				break
		break
			

		print('Epoch {} \tTotal_loss: {} \tce_loss: {} \tKLD_loss{}'.format(epoch, np.mean(total_loss_list[-(len(total_loss_list)//epoch):]), np.mean(ce_loss_list[-(len(ce_loss_list)//epoch):]), np.mean(KLD_loss_list[-(len(KLD_loss_list)//epoch):]) ))
	
	##Return the loss lists, so we can save a plot where the loss is showed.
	return (total_loss_list, ce_loss_list, KLD_loss_list)







 	
