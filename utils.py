#import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from sklearn.decomposition import PCA
import pandas as pd
import pickle



def pca_encode(model, dataset, labels, outfile):
	"""
	Function for plotting encoded data
	DOES NOT AVERAGE OVER SAME CONTIGS.
	"""
	
	##Encode the whole dataset in small parts (memory limits)
	for i in range(0, dataset.shape[0], 500):
		if i == 0:
			encoded_vector = np.array(model.encode(torch.from_numpy(dataset[i:i+500]).float())[0].cpu().detach().numpy() )
		else:
			encoded_vector = np.append(encoded_vector, model.encode(torch.from_numpy(dataset[i:i+500]).float())[0].cpu().detach().numpy(), axis=0 )
	
	targets = []
	for line in labels:
		targets.append(line.split()[0])
	
	##Instantiate 2-dimenstional PCA
	PCA_latent_space = PCA(2)
	PCA_latent_space.fit(encoded_vector)
	explained_variance = PCA_latent_space.explained_variance_ratio_
	latent_PCA = PCA_latent_space.transform(encoded_vector)
	
	##Create a numpy array with all the unique labels
	unique_targets = np.unique(targets).astype(str)
	latent_dict = dict()
	for target in unique_targets:
		latent_dict[target] = []

	##Create a pandas dataframe with 3 columns (PCA1, PCA2, target)
	df_PCA = pd.DataFrame(latent_PCA)
	df_PCA.columns = ['PCA 1', 'PCA 2']
	df_PCA['target'] = targets
	np_PCA = np.array(df_PCA)

	##Plot PCA datapoints and save to file specified by function input.
	for i in range(len(np_PCA)):
		latent_dict[np_PCA[i,2]].append(np_PCA[i,0:2])
	latent_dict["variance"] = explained_variance

	##Saving dictionary to a pickle, to SCP to own computer and work with data
	with open(outfile, "wb") as f:
		pickle.dump(latent_dict, f)
	
	"""	
	fig = plt.figure()
	for key, value in latent_dict.items():
		plt.scatter(np.array(value)[:,0], np.array(value)[:,1], label=key)
	plt.xlabel('PCA 1, variance: ' + str(explained_variance[0]))
	plt.ylabel('PCA 2, variance: ' + str(explained_variance[1]))
	plt.title('Principal Component Analysis', fontsize=15)
	plt.legend()
	fig.savefig(outfile+'.pdf', bbox_inches='tight')
	"""

	return None



def pca_avg_encode(model, dataset, labels, outfile):
	print('Using function pca_avg_encode')

	##Encode the whole dataset in small parts (memory limits)
	for i in range(0, dataset.shape[0], 500):
		if i == 0:
			encoded_vector = np.array(model.encode(torch.from_numpy(dataset[i:i+500]).float())[0].cpu().detach().numpy() )
		else:
			encoded_vector = np.append(encoded_vector, model.encode(torch.from_numpy(dataset[i:i+500]).float())[0].cpu().detach().numpy(), axis=0 )
	
	##Make a list of targets(OTU) and contigs
	targets = []
	contigs = []
	for line in labels:
		line_split = line.split()
		targets.append(line_split[0])
		contigs.append(line_split[1])
	targets, contigs = np.array(targets), np.array(contigs)	
	

	##Instantiate 2-dimenstional PCA
	PCA_latent_space = PCA(2)
	PCA_latent_space.fit(encoded_vector)
	explained_variance = PCA_latent_space.explained_variance_ratio_
	latent_PCA = PCA_latent_space.transform(encoded_vector)
	
	##Create a numpy array with all the unique labels
	unique_targets = np.unique(targets).astype(str)
	latent_dict = dict()
	for target in unique_targets:
		latent_dict[target] = []

	##Create a pandas dataframe with 3 columns (PCA1, PCA2, target)
	#print(latent_PCA.shape)
	df_PCA = pd.DataFrame(latent_PCA)
	df_PCA.columns = ['PCA 1', 'PCA 2']
	df_PCA['target'] = targets
	np_PCA = np.array(df_PCA)

	##Iterate through contigs and OTU and have index for when contig is the same. Use this to compute the average in the latent space.
	index_start, index_end = 0, 0
	np_avg_PCA = None
	current_contig = contigs[0]
	df_avg_labels = []
	for i in range(len(targets)):
		if contigs[index_start] == contigs[index_end]:
			index_end += 1
		else:
			df_avg_labels.append(targets[index_start])
			if np_avg_PCA is None:
				np_avg_PCA = np.array([[np.mean(np_PCA[index_start:index_end, 0]), np.mean(np_PCA[index_start:index_end, 1])]])
			else:
				np_avg_PCA = np.concatenate((np_avg_PCA, np.array([[np.mean(np_PCA[index_start:index_end, 0]), np.mean(np_PCA[index_start:index_end, 1])]])))					
			index_start = index_end
			index_end += 1

	##Create a new DF for the new averaged data
	print(np_avg_PCA.shape)
	df_avg_PCA = pd.DataFrame(np_avg_PCA)
	df_avg_PCA.columns = ["PCA 1", "PCA 2"]
	df_avg_PCA["target"] = df_avg_labels
	np_avg_PCA = np.array(df_avg_PCA)

	##Create dict for easy plotting
	PCA_dict = dict()
	for target in np.unique(df_avg_labels).astype(str):
		PCA_dict[target] = []
	for i in range(len(np_avg_PCA)):
		PCA_dict[np_avg_PCA[i,2]].append(np_avg_PCA[i,0:2])
	PCA_dict["variance"] = explained_variance

	##Saving dictionary to a pickle, to SCP to own computer and work with data
	with open(outfile, "wb") as f:
		pickle.dump(PCA_dict, f)


	##Plot figure and save it to a pdf
	"""
	fig = plt.figure()
	for key, value in PCA_dict.items():
		plt.scatter(np.array(value)[:,0], np.array(value)[:,1], label=key)
	plt.xlabel('PCA 1, variance: ' + str(explained_variance[0]))
	plt.ylabel('PCA 2, variance: ' + str(explained_variance[1]))
	plt.title('Principal Component Analysis', fontsize=15)
	plt.legend()
	fig.savefig(outfile+'.pdf', bbox_inches='tight')
	"""

	return None
			
		









		
