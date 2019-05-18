##Script for creating a numpy memmap for all the contig tensors.

import numpy as np
import os

os.chdir('/home/projects/cpr_10006/projects/cnn_vamb/process_contigs/contig_tensors')

memmap_file = np.memmap('/home/projects/cpr_10006/projects/cnn_vamb/process_contigs/contig_tensors/memmap_tensor', dtype=np.int, mode='w+', shape=(752480, 4, 2000))

npy_files = os.listdir()

for i in range(1, len(npy_files)):
	if i+1 == len(npy_files):
		memmap_file[1000*(i-1):, :, :] = np.load('numpy_tensor'+str(i)+'.npy')
	else:
		memmap_file[1000*(i-1):1000*i,:,:] = np.load('numpy_tensor'+str(i)+'.npy')

memmap_file.flush()
