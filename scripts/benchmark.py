import vamb

def benchmark(cluster_directory):
	print('Benchmarking')

	with open("/home/projects/cpr_10006/projects/cnn_vamb/cnn/reference.tsv") as filehandle:
	    reference = vamb.benchmark.Reference.from_file(filehandle)

	with open(str(cluster_directory)+'/cluster.tsv') as filehandle:
	    clusters = vamb.cluster.read_clusters(filehandle)

	binning = vamb.benchmark.Binning(clusters, reference)
	bin_file = open(str(cluster_directory)+'/bins.tsv', 'w')
	binning.print_matrix(file=bin_file)
	bin_file.close()

	log_file = open(str(cluster_directory)+'/bins_info.tsv', 'w')
	print('Bin name', 'DNA in bin', 'number of contigs', 'recall', 'precision', sep='\t', file=log_file)
	for bin_name in binning.bin_precision:
		print(bin_name, binning.bin_precision[bin_name]['length'], len(binning.bin_precision[bin_name]['contigs']), binning.bin_precision[bin_name]['recall'], binning.bin_precision[bin_name]['precision'], sep='\t', file=log_file)
		#print(bin_name, binning.breadthof[bin_name])
	return None
run = input('Write directory: ')

benchmark('/home/projects/cpr_10006/projects/cnn_vamb/cnn/runs/'+run)

