# Convolutional neural networks for unsupervised learning of patterns in DNA sequences


**Next-Generation-Sequencing has allowed scientist to develop a high-throughput method
for sequencing DNA with low cost and high precision. This has led to advances in
bioinformatics and especially metagenomics. In metagenomics the goal is to identify
and reconstruct microbial species directly from environmental samples. This allows
scientist to discover new organisms that could potentially contain new biological
pathways with great interest for many industries.
In this report, the possibiliies of combining deep learning techniques and metagenomics was discovered, by creating a Convolutional Variational AutoEncoder that
was trained directly on contigs to cluster sequencing data based on taxonomy. It was
found that using Convolutional layers in the Variational AutoEncoder had a detrimental effect on the results, because of the requirement that each input sequence
had to be the same length, which forced the contigs to be fragmented. This showed
to weaken the signal in the data to such a degree that the model was not able to
create meaningful clusters.**
