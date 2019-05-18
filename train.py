import torch
import numpy as np
from torch.utils.data import DataLoader


def train_model(model, dataset, epochs, batch_size, lr, cuda, gumbel):
	'''
	Function for training a model. 
	'''

	model.train()
	optim = torch.optim.Adam(model.parameters(), lr=lr)
	total_loss_list = []
	ce_loss_list = []
	KLD_loss_list = []
	
	if gumbel:
		tau0 = 1
		anneal_rate = 0.004
		min_tau = 0.5



	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	print('Starting to train!')

	for epoch in range(1, epochs+1):
		total_loss_item = 0
		ce_loss_item = 0
		KLD_loss_item = 0	
		#batch_idx = np.random.permutation(dataset.shape[0])

		for i_batch, batch in enumerate(dataloader):
			if cuda:
				batch = batch.float().cuda()
			else:
				batch = batch.float()
			
			reconstructed_x, mu, logvar = model.forward(batch)

			##Calculating Reconstruction loss and KLD loss, from method in model Class.
			if gumbel:
				tau = np.maximum(tau0*np.exp(-anneal_rate*epoch), min_tau)
				total_loss, ce_loss, KLD_loss = model.calc_loss(reconstructed_x, batch, mu, logvar, tau=tau)
			else:
				total_loss, ce_loss, KLD_loss = model.calc_loss(reconstructed_x, batch, mu, logvar)
			
			optim.zero_grad()
			total_loss.backward()
			optim.step()
			
			##Append loss to list to visualize model loss improvement.
			total_loss_item += total_loss.item()
			ce_loss_item += ce_loss.item()
			KLD_loss_item += KLD_loss.item()

		total_loss_list.append(total_loss_item/i_batch)
		ce_loss_list.append(ce_loss_item/i_batch)
		KLD_loss_list.append(KLD_loss_item/i_batch)
			

		print('Epoch {} \tTotal_loss: {} \tce_loss: {} \tKLD_loss: {}'.format(epoch, ce_loss_list[-1]+KLD_loss_list[-1], ce_loss_list[-1], KLD_loss_list[-1]))
	
	##Return the loss lists, so we can save a plot where the loss is showed.
	return (total_loss_list, ce_loss_list, KLD_loss_list)







 	
