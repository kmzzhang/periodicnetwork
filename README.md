Code associated with the paper:
  > Classification of periodic variables with novel cyclic-permutation invariant neural networks\
  > Keming Zhang & Joshua Bloom\
  > currently under review

This repository contains pytorch implementations of the cyclic-permutation invariant networks used in our study.
It also contains code to reproduce our main results.

Neural network implementations can be found under ./model/

**Description of files**:\
**train.py**: train neural networks on variable star light curve datasets\
**ppmnist.py**: train neural networks on the periodic permuted MNIST task\
**run_variablestar.sh**: commmands for reproducing our variable star experiements\
**run_ppmnist.sh**: commmands for reproducing our PP-MNIST experiements\
**data/download.sh**: script to download the datasets\
**trained_models.tar**: trained models of Table 1 of the paper

We provide two options for reproducing our results. To test on our provided trained models, decompress
**trained_models.tar** into the ./results first. Alternatively, you may opt to train these models from stretch
simply by removing the --test option in the **run_variablestar.sh** commands. If you do so, we suggest that a new conda
environment be created from environment.yml to replicate the identical software environment. 

You may use the --ngpu and the --njob options to facilitate parallel training/testing
with multiple gpus or processes. If GPU device is not available for your device, you can still specify --njob
for parallel training/testing. The code automatically detects the availability of GPUs.

The variable star light curve datasets used in this study have been uploaded to zenodo, and can be downloaded using the
script in ./data/download.sh. These datasets have been constructed from publicly available databases. 
If you use this dataset, please cite the original papers, the citation of which can be found in 
https://zenodo.org/record/3903015

