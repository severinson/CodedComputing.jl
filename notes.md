## Genome data

### Data format

For PCA, and to plot populations separately, we will use data from the 1000 genomes project. We need two kinds of files: VCF files and PED files.

VCF files record genetic variations, i.e., how genetic samples (e.g., taken from humans, with each sample corresponding to one individual). Each of these possible differences is referred to as a variant. The 1000 genomes VCF files each correspond to one chromosome. Humans have 23 chromosomes, 22 of which are shared by all humans (except in case of certain genetic conditions), and one which corresponds to sex (each human has either a X or Y chromosome).

For each file, each line corresponds to a possible variant, i.e., a possible difference in the sample genome from the reference genome. The first few columns store information about the variant, such as its position in the chromosome, what the reference genome contains at that position, and how the variant may differ. The final columns correspond to genome samples (each sample may correspond to an individual), and the entries of these columns denote how that particular sample differs from the reference (there may be several possibilities).

The PED file stores information about each sample, such as the population it is taken from.

### Workflow

We will compute PCA on a binary matrix, for which each row corresponds to a sample and each column to a position in the genome. For each entry, a 1 indicates that sample differs from the reference (we don't care how it differs).

The strategy used is based on 
https://github.com/bwlewis/1000_genomes_examples
http://bwlewis.github.io/1000_genomes_examples/notes.html
bwlewis.github.io/1000_genomes_examples/PCA.html
bwlewis.github.io/1000_genomes_examples/PCA_overview.html

1. Download a VCF file from the 1000 genomes project for some chromosome ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/
2. Convert that file into a CSV file using the parse.c program in the 1000genomes folder
3. Read that CSV file into a Julia DataFrame
4. Create a sparse matrix from the DataFrame by using the second column of the DataFrame as row indices and the first as column indices
5. Compute the principal components of that matrix
6. Project the sample data onto the principal components
7. Match each sample to the population it is taken from
8. Plot the projected samples separately for each population