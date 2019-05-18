import re
import os

bad_DNA_file = open('/home/projects/cpr_10006/projects/cnn_vamb/process_contigs/bad_DNA_entries_10000.txt')
bad_DNA_set = set()
for line in bad_DNA_file:
	entry = line.split()[-1]
	entry = entry.split(':')
	interval = re.findall('\d+', entry[-1])
	entry_append = str(entry[0][1:]) + '\t' + str(interval[-2]) + '\t' + str(interval[-1])
	bad_DNA_set.add(entry_append)
bad_DNA_file.close()
#print(bad_DNA_set)
#print(len(bad_DNA_set)) 
	
reference_dict = dict()
reference_tsv_file = open('/home/projects/cpr_10006/projects/cnn_vamb/jakni/reference.tsv')
for line in reference_tsv_file:
	line = line.split()
	reference_dict[line[1]] = line[0]

contig_interval_file = open('/home/projects/cpr_10006/projects/cnn_vamb/process_contigs/genome.windows_10000.bed')

output_file = open('labels_10000.txt', 'w')
bad_dna_count = 0
short_dna_count = 0
count = 0
for line in contig_interval_file:
	line_split = line.split()
	if line.rstrip() in bad_DNA_set:
		#print('Was found in bad_DNA file!')
		bad_dna_count += 1
	elif abs(int(line_split[-2]) - int(line_split[-1])) < 10000:
		#print('Too short DNA was found!')
		short_dna_count += 1
	else:
		print(reference_dict[line_split[0]] + '\t' + line_split[0] + '\t' + str(line_split[-2]) +'-'+str(line_split[-1]), file=output_file)
		count += 1

contig_interval_file.close()
output_file.close()
print('{} entries has been checked'.format(bad_dna_count+short_dna_count+count))
print('{} was bad DNA'.format(bad_dna_count))
print('{} was short DNA'.format(short_dna_count))
print('{} was good DNA and is put into the file labels.txt'.format(count))




	

