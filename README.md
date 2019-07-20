# Convolutional neural networks for unsupervised learning of patterns in DNA sequences

This report is the product of a 15 ECTS Bachelor thesis for the completion of my
BSc. Eng. Biotechnology at The Technical University of Denmark (DTU). The
project was conducted from the 1st of february to the 5th of June 2019 at the facilities of Novo Nordisk Center for Protein Research (CPR) in cooperation with the
Rasmussen group that focuses on Human Proteome Variation. The external supervisor from CPR was group leader Simon Rasmussen.

In short, this project was about using a Varitional AutoEncoder to cluster DNA sequences based on taxonomy (What kind of group the organism belongs to). This is a problem with great interest for bioinformaticians as it will enable fast screening of different environmental samples. An example of this, could be within the medical field, where the human microbiome has been shown to have a great influence on our health. If we can develop quick and cheap solutions to classifying the organisms living in the gut, we can use this information to give more impactful treatments in the future.



### Abstract:

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

The report can be found [here](Bachelor.pdf). It contains all the methods used and the results generated from the project.
