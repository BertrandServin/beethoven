# Beethoven

This repository contains a python package to recontruct queen genotypes from
pool sequence data of honeybee hives. This implements the methods described in
the paper:

From group to individual - Genotyping by pool sequencing eusocial colonies
Sonia E Eynard, Alain Vignal, Benjamin Basso, Yves Le Conte, Axel Decourtye,
Lucie Genestout, Emmanuelle Labarthe, Fanny Mondet, Kamila Tabet, Bertrand Servin
bioRxiv 2021.11.08.467442; doi: https://doi.org/10.1101/2021.11.08.467442

## Install the package

``` shell
pip install .
```
should do it.

## Run the scripts

Two scripts are provided. The first `qg_pool` allows to estimate the admixture
proportions of the queen from the pool seq data

``` shell
qg_pool -h
```
will give you the syntax to run the program.

The second `genoqueen_hom` is run like this:

genoqueen_hom dir_in depth.txt count_ref.txt n_col_snpid ncpu oprefix batch_len

- dir_in is the directory containing the input files
- depth.txt and count_ref.txt contain the depth and counts information for a set
  colonies. An example of 3 colonies with 1 SNP

``` shell
$ cat depth.txt
rs12 1 290484 30 35 20
$ cat count_ref.txt
rs12 1 290484 15 5 18
```
- n_col_snpid is the number of columns to skip before reading counts/depth (3 in
  the example above)
- ncpu : number of CPUs to use
- oprefix : output prefix
- batch_len : size of batches that run in parallel (here 1000 SNPs).
