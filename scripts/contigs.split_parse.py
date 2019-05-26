##Script for saving splitted contigs of size 2000 into tensors of size [1000, 4, 2000].

import torch
import os
import re
import numpy as np

##Initialization before creating contig-tensors
OneHot_dict = {'A':np.array([[1,0,0,0]]), 'T':np.array([[0,1,0,0]]), 'G':np.array([[0,0,1,0]]), 'C':np.array([[0,0,0,1]])}
OneHot_matrix = np.array([])
OneHot_tensor = None
split_contig_file_name = '/home/projects/cpr_10006/projects/cnn_vamb/split_contigs/contigs.split_10000.fasta'
infile = open(split_contig_file_name, 'r')
bad_DNA_file_name = '/home/projects/cpr_10006/projects/cnn_vamb/process_contigs/bad_DNA_entries_10000.txt'
bad_DNA_file = open(bad_DNA_file_name, 'w')
read_flag = True
count_num_entries = 0
count_num_saves = 0
for line in infile:
	
	##Save the contig-tensor if it contains 1000 contig-fragments
	if count_num_entries % 1000 == 0 and len(OneHot_matrix) > 0:
		OneHot_matrix = np.array([OneHot_matrix.T])
		#print(OneHot_matrix.shape)		
		#print(OneHot_tensor.shape)
		OneHot_tensor = np.concatenate((OneHot_tensor, OneHot_matrix))
		print(OneHot_tensor.shape)
		OneHot_matrix = np.array([])
		np.save('/home/projects/cpr_10006/projects/cnn_vamb/process_contigs/contig_tensors_10000/numpy_tensor' + str(count_num_saves+1), OneHot_tensor)		
		count_num_saves += 1
		print('Parsed {} entries, {} was saved into file: {}'.format(1000*count_num_saves, OneHot_tensor.shape[0], 'numpy_tensor' +str(count_num_saves)))
		OneHot_tensor = None
	##Check if new contig-fragment has desired length
	if line.startswith('>'):
		#count_num_entries += 1
		line_header = line.rstrip()
		interval = re.findall('\d+', line_header)
		DNA_length = abs(int(interval[-1]) - int(interval[-2]))
		if DNA_length >= 10000:
			count_num_entries += 1
		read_flag = False
		DNA_correct = True
		##Append this DNA to the contig-tensor
		if len(OneHot_matrix) > 0:
			OneHot_matrix = np.array([OneHot_matrix.T])
			if OneHot_tensor is None:
				OneHot_tensor = OneHot_matrix
				OneHot_matrix = np.array([])
			else:
				OneHot_tensor = np.concatenate((OneHot_tensor, OneHot_matrix))
				OneHot_matrix = np.array([])
	##If fragment length is big enough and DNA is A,T,C og G
	if read_flag and DNA_length >= 10000 and DNA_correct:
		for base in line.rstrip():
			if len(OneHot_matrix) == 0:
				if base in OneHot_dict:
					OneHot_matrix = OneHot_dict[base]
				else:
					OneHot_matrix = np.array([])
					DNA_correct = False
					print('The base {} was found in: {}'.format(base, line_header), file=bad_DNA_file)
					count_num_entries -= 1
					break
			else:
				if base in OneHot_dict:
					OneHot_matrix = np.concatenate((OneHot_matrix, OneHot_dict[base]))
				else:
					OneHot_matrix = np.array([])
					DNA_correct = False
					print('The base {} was found in: {}'.format(base, line_header), file=bad_DNA_file)
					count_num_entries -= 1
					break
	read_flag = True

##Save last contig-tensor aswell
if len(OneHot_matrix) > 0:
	OneHot_matrix = np.array([OneHot_matrix.T])
	if OneHot_tensor is None:
		OneHot_tensor = OneHot_matrix
		OneHot_matrix = np.array([])
	else:
		OneHot_tensor = np.concatenate((OneHot_tensor, OneHot_matrix))
		OneHot_matrix = np.array([])
np.save('/home/projects/cpr_10006/projects/cnn_vamb/process_contigs/contig_tensors_10000/numpy_tensor' + str(count_num_saves+1), OneHot_tensor)
print('Parsed {} entries, {} was saved into file: {}'.format(1000*count_num_saves, OneHot_tensor.shape[00
], 'numpy_tensor' +str(count_num_saves)))
bad_DNA_file.close()
infile.close()





