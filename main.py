import argparse
import torch
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import os
from model import CVAE
from train import train_model
from utils import *


parser = argparse.ArgumentParser('CVAE')


parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1028)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--latent_size', type=int, default=30)
parser.add_argument('--ce_weight', type=float, default=1)
parser.add_argument('--KLD_weight', type=float, default=1)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--channels', type=str, default='50')
parser.add_argument('--kernels', type=str, default='20')
parser.add_argument('--strides', type=str, default='20')

parser.add_argument('--train', action='store_true')
parser.add_argument('--load', action='store_true')
parser.add_argument('--load_path', type=str, default='')

parser.add_argument('--save', type=str, default='')




if __name__ == '__main__':
	args = parser.parse_args()

	##Load dataset and labels
	dataset = np.memmap('/home/projects/cpr_10006/projects/cnn_vamb/cnn/data/memmap_tensor', dtype=np.int, mode='r', shape=(752480, 4, 2000))
	labels = open('/home/projects/cpr_10006/projects/cnn_vamb/cnn/data/labels.txt', 'r')

	channels = list(map(int, args.channels.split('_')))
	kernels = list(map(int, args.kernels.split('_')))
	strides = list(map(int, args.strides.split('_')))

	#The model takes the following inputs: (latent_size, ce_weight, KLD_weight)
	cvae = CVAE(args.latent_size, args.ce_weight, args.KLD_weight, channels, kernels, strides)
	
	##Check for cuda	
	if args.cuda:
		cvae = cvae.cuda()
	else:
		cvae = cvae.float()
	
	##Train model if in arguments
	if args.train:
		loss_list = train_model(cvae, dataset, args.epochs, args.batch_size, args.lr, args.cuda)
	else:
		print('Not training. train_state is {}'.format(args.train))
	
	##Save model into directory specified by user
	if len(args.save) > 0:
		os.mkdir(os.getcwd()+'/' + str(args.save))
		torch.save(cvae, './'+str(args.save)+'/modellos')
		info_file = open('./'+str(args.save)+'/log_file', 'w')
		info_file.write('### File containing specs for model ###\n')
		for arg, value in vars(args).items():
			print(arg, value, sep='\t', file=info_file)
		print('### MODEL PARAMETERS ###', file=info_file)
		print(cvae, file=info_file)

		##If model is trained, and a directory to save has been specified then plot the loss.
		if args.train:

			##Call function from utils.py to PCA plot encoded samples
			#pca_encode(cvae, dataset, labels, './'+str(args.save)+'/Encode_PCA.pkl')
			pca_avg_encode(cvae, dataset, labels, './'+str(args.save)+'/Encode_avg_PCA.pkl')
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
			

			
















		
	
