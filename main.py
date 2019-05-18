import argparse
import torch
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import os
from model import CVAE
from train import train_model
from utils import *
#from benchmark import *


parser = argparse.ArgumentParser('CVAE')


parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1028)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--latent_size', type=int, default=30)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gumbel', action='store_true')
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--channels', type=str, default='50')
parser.add_argument('--kernels', type=str, default='20')
parser.add_argument('--strides', type=str, default='20')
parser.add_argument('--padding', type=str, default='')

parser.add_argument('--encode', type=str, default='')
parser.add_argument('--train', action='store_true')
parser.add_argument('--load', type=str, default='')
parser.add_argument('--cluster', type=str, default='')
parser.add_argument('--benchmark', type=str, default='')

parser.add_argument('--save', type=str, default='')




if __name__ == '__main__':
	args = parser.parse_args()


	##Load dataset and labels
	#dataset = np.memmap('/home/projects/cpr_10006/projects/cnn_vamb/cnn/data/memmap_tensor', dtype=np.int, mode='r', shape=(752480, 4, 2000))
	dataset = np.load('/home/projects/cpr_10006/projects/cnn_vamb/cnn/data/combined_tensor_10000.npy')
	labels = open('/home/projects/cpr_10006/projects/cnn_vamb/cnn/data/labels_10000.txt', 'r')

	##Create list of convolution parameters (Create try/except if not equal length?)
	channels = list(map(int, args.channels.split('_')))
	kernels = list(map(int, args.kernels.split('_')))
	strides = list(map(int, args.strides.split('_')))
	if len(args.padding) == 0:
		padding = np.zeros(len(channels))
	else:
		padding = list(map(int, args.padding.split('_')))
	
	cuda_flag = args.cuda
	
	##Check for cuda and create/load model	
	if cuda_flag:
		if len(args.load) > 0:
			cvae = torch.load(args.load+'/model')
			cvae.load_state_dict(torch.load(args.load+'/state_dict'))
			print('Model loaded')
		else:
			cvae = CVAE(args.latent_size, args.alpha, args.dropout, channels, kernels, strides, padding, args.cuda, args.gumbel, dataset.shape[-1]).float().cuda()
	else:
		if len(args.load) > 0:
			cvae = torch.load(args.load+'/model', map_location='cpu')
			cvae.load_state_dict(torch.load(args.load+'/state_dict', map_location='cpu'))
			print('Model loaded')
		else:
			cvae = CVAE(args.latent_size, args.alpha, args.dropout, channels, kernels, strides, padding, args.cuda, args.gumbel, dataset.shape[-1]).float()
	
	##Train model if in arguments
	if args.train:
		loss_list = train_model(cvae, dataset, args.epochs, args.batch_size, args.lr, args.cuda, args.gumbel)
	else:
		print('Not training. train_state is {}'.format(args.train))
	
	##Save model into directory specified by user
	if len(args.save) > 0:
		##If we have not created a directory for the model, then do it
		if not os.path.isdir(os.getcwd()+'/' + str(args.save)):
			os.mkdir(os.getcwd()+'/' + str(args.save))
			info_file = open('./'+str(args.save)+'/log_file', 'w')
			info_file.write('### File containing specs for model ###\n')
			for arg, value in vars(args).items():
				print(arg, value, sep='\t', file=info_file)
			print('### MODEL PARAMETERS ###', file=info_file)
			print(cvae, file=info_file)
		
		##Check if we have saved a model allready. If we have, then remove it and save the newly trained model
		if os.path.isfile('./'+str(args.save)+'/model'):
			os.remove('./'+str(args.save)+'/model')
			os.remove('./'+str(args.save)+'/state_dict')
			torch.save(cvae, './'+str(args.save)+'/model')
			torch.save(cvae.state_dict(), './'+str(args.save)+'/state_dict' )
		else:
			torch.save(cvae, './'+str(args.save)+'/model')
			torch.save(cvae.state_dict(), './'+str(args.save)+'/state_dict' )
		
		##If model is trained, and a directory to save has been specified then plot the loss.
		if args.train:

			##Call function from utils.py to PCA plot encoded samples
			#pca_encode(cvae, dataset, labels, './'+str(args.save)+'/Encode_PCA.pkl')
			if os.path.isfile('./'+str(args.save)+'/Encode_avg_PCA.pkl'):
				os.remove('./'+str(args.save)+'/Encode_avg_PCA.pkl')
				pca_avg_encode(cvae, dataset, labels, args.cuda, './'+str(args.save)+'/Encode_avg_PCA.pkl')
			else:
				pca_avg_encode(cvae, dataset, labels, args.cuda, './'+str(args.save)+'/Encode_avg_PCA.pkl')
			if not os.path.isfile('./'+str(args.save)+'/loss_plot.pkl'):
				with open('./'+str(args.save)+'/loss_plot.pkl','wb') as f:
					pickle.dump(loss_list, f)
			'''
			fig = plt.figure()
			plt.plot(range(1, len(loss_list[0])+1), loss_list[0], color='r')
			plt.plot(range(1, len(loss_list[1])+1), loss_list[1], color='b')
			plt.plot(range(1, len(loss_list[2])+1), loss_list[2], color='g')
			plt.legend(['Total loss', 'CE_loss', 'KLD_loss'])
			plt.title('Loss plt')
			plt.xlabel('Epochs')
			plt.ylabel('Loss')
			fig.savefig('./'+str(args.save)+'/Loss_plot.pdf', bbox_inches='tight')
			'''
	##If the user only wants to encode with a loaded model
	if len(args.encode) > 0:
		if os.path.isfile('./'+str(args.encode)+'/Encode_avg_PCA.pkl'):
			os.remove('./'+str(args.encode)+'/Encode_avg_PCA.pkl')
			pca_avg_encode(cvae, dataset, labels, args.cuda, './'+str(args.encode)+'/Encode_avg_PCA.pkl')
		else:
			pca_avg_encode(cvae, dataset, labels, args.cuda, './'+str(args.encode)+'/Encode_avg_PCA.pkl')
	
	##If the user wants to cluster with a loaded/trained model
	if len(args.cluster) > 0:
		cluster_encoding(cvae, dataset, labels, args.cuda, './'+str(args.cluster)+'/cluster.tsv')

					

			
















		
	
