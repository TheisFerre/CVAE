import vamb

with open('/home/projects/cpr_10006/projects/cnn_vamb/contigs.fna', 'rb') as file:
    fastadict = vamb.vambtools.loadfasta(file)

cluster_iterator_location = input('Write name of directory where cluster.tsv file is located: ')
cluster_iterator = open(str(cluster_iterator_location)+'/cluster.tsv', 'r')
clusters = dict(cluster_iterator)

bindir = str(cluster_iterator_location)+'/write_bins'
vamb.vambtools.write_bins(bindir, clusters, fastadict, maxbins=500)




'''
def splitclusters(clusters, sep):
    # First split by sample
    split = dict()
    for n, (clustername, contigs) in enumerate(clusters.items()):
        for contig in contigs:
            samplename = contig.partition(sep)[0]
            newclustername = str(n) + '_' + samplename
            if newclustername in split:
                split[newclustername].add(contig)
            else:
                split[newclustername] = {contig}
    return split

def filterclusters(clusters, lengthof):
    # Now filter away the small bins
    filtered_bins = dict()
    for medoid, contigs in clusters.items():
        binsize = sum(lengthof[contig] for contig in contigs)
    
        if binsize >= 200000:
            filtered_bins[medoid] = contigs
    
    return filtered_bins
        
lengthof = dict(zip(contignames, lengths))
filtered_bins = filterclusters(splitclusters(clusters, '_'), lengthof)
print('Number of bins before splitting and filtering:', len(clusters))
print('Number of bins after splitting and filtering:', len(filtered_bins))
'''
