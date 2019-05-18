###Script for counting if every entry has been parsed.
import re
import os
import numpy as np

bad_DNA_count = 0
bad_DNA_file = open('bad_DNA_entries.txt', 'r')
for line in bad_DNA_file:
	bad_DNA_count += 1

bad_DNA_file.close()

short_DNA_count = 0
contig_header_file = open('genome.windows.bed', 'r')
for line in contig_header_file:
	interval = re.findall('\d+', line.rstrip())
	DNA_length = abs(int(interval[-1]) - int(interval[-2]))
	if DNA_length < 2000:
		short_DNA_count += 1
contig_header_file.close()
os.chdir('/home/projects/cpr_10006/projects/cnn_vamb/process_contigs/contig_tensors')

valid_DNA_count = 0
for numpy_file in os.listdir():
	matrix = np.load(numpy_file)
	valid_DNA_count += matrix.shape[0]

print('Contigs with bad DNA: {}'.format(bad_DNA_count))
print('Contigs with length > 2000: {}'.format(short_DNA_count))
print('Valid contigs: {}'.format(valid_DNA_count))

print('Total number of processed contigs: {}'.format(bad_DNA_count+short_DNA_count+valid_DNA_count))
