##Script for combining contig tensors to a single big tensor. This could have been merged together with the script that creates the contig-tensors to one big script.

import os
import numpy as np

os.chdir('/home/projects/cpr_10006/projects/cnn_vamb/process_contigs/contig_tensors_10000')
files = os.listdir()
prefix = 'numpy_tensor'

##Goes through the directory and concatenates the files into a single big .npy file
for i in range(1, len(os.listdir())+1):
	file_name = prefix+str(i)+'.npy'
	if i == 1:
		combined_tensor = np.load(file_name)
		print('Tensor {} has been parsed.'.format(i))
		print(combined_tensor.shape)
	else:
		combined_tensor = np.concatenate((combined_tensor, np.load(file_name)))
		print('Tensor {} has been parsed.'.format(i))
		print(combined_tensor.shape)

np.save('combined_tensor_10000', combined_tensor)
