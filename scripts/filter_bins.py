import os
directory = input('write directory where the file is: ')


input_file = directory+'/bins_info.tsv'

in_file = open(input_file, 'r')

bin_dict = dict()

thresholds = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

for line in in_file:
	if line[0:7] == 'cluster':
		if int(line.split()[1]) >= 100000:
			bin_dict[line.split()[0]] = [line.split()[-2], line.split()[-1]]
in_file.close()

outfile = open(directory+'/bins_info_filtered.tsv', 'w')
print('', *thresholds, sep='\t', file=outfile)

for recall in thresholds:
	if recall != 0.3:
		print(file=outfile)
	for precision in thresholds:
		if precision == 0.3:
			print(recall, end='\t', file=outfile)
		count = 0
		for _, values in bin_dict.items():
			if float(values[0]) >= recall and float(values[1]) >= precision:
				count += 1
		print(count, end='\t', file=outfile)
outfile.close()
	
	


