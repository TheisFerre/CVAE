#import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from sklearn.decomposition import PCA
import pandas



def pca_encode(model, dataset, labels, outfile):
	"""
	Function for plotting encoded data
	DOES NOT AVERAGE OVER SAME CONTIGS.
	"""
	
	##Encode the whole dataset in small parts (memory limits)
	for i in range(0, data_set.shape[0], 500):
		if i == 0:
			encoded_vector = np.array(model.encode(torch.from_numpy(dataset[i:i+500]))[0].cpu().detach().numpy() )
		else:
			encoded_vector = np.append(encoded_vector, model.encode(torch.from_numpy(dataset[i:i+500]))[0].cpu().detach().numpy(), axis=0 )
	
	##Instantiate 2-dimenstional PCA
	PCA_latent_space = PCA(2)
	PCA_latent_space.fit(encoded_vector)
	explained_variance = PCA_latent_space.explained_variance_ratio_
	latent_PCA = PCA_latent_space.transform(encoded_vector)
	
	##Create a numpy array with all the unique labels
	unique_targets = np.unique(labels).astype(str)
	latent_dict = dict()
	for label in unique_targets:
		latent_dict[label] = []

	##Create a pandas dataframe with 3 columns (PCA1, PCA2, LABEL)
	df_PCA = pd.DataFrame(latent_PCA)
	df_PCA.columns = ['PCA 1', 'PCA 2']
	df_PCA['label'] = labels
	np_PCA = np.array(df_PCA)

	##Plot PCA datapoints and save to file specified by function input.
	for i in range(len(np.PCA)):
		latent_dict[np_PCA[i,2]].append(np_PCA[i,0:2])
	fig = plt.figure()
	for key, value in latent_dict.items():
		plt.scatter(np.array(value)[:,0], np.array(value)[:,1], label=key)
	plt.xlabel('PCA 1, variance: ' + str(explained_variance[0]))
	plt.ylabel('PCA 2, variance: ' + str(explained_variance[1]))
	plt.title('Principal Component Analysis', fontsize=15)
	plt.legend()
	fig.savefig(outfile+'.pdf', bbox_inches='tight')

	return None









		
