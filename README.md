# Cyclic-Permutation Invariant Neural Networks

Code associated with the paper:
  > **Classification of Periodic Variable Stars with Novel Cyclic-Permutation Invariant Neural Networks**\
  > Keming Zhang & Joshua Bloom\
  > Submitted to MNRAS\
  > [arXiv:2011.01243](https://arxiv.org/abs/2011.01243)\
  > Accepted to ICLR 2020: FSAI (spotlight talk)\
  > Accepted to NeurIPS 2020: ML4PS

This repository contains pytorch implementations of the cyclic-permutation invariant networks used in our study.
It also contains code to reproduce our main results.

Neural network implementations can be found under ./model/

## Reproducing our results

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

## Adapting to your own data

Code for iTCN and iResNet can be found under ``model``. To implement cyclic-permutation invariance in your custom CNN
archetecture written in pytorch, add the ``wrap`` module under ``model/padding.py`` before any nn.Conv1D module and 
remove existing padding.

You might find ``times_to_lags`` under ``util.py`` useful for transforming times to time intervals which accounts for
periodicity.
```python
def times_to_lags(x, p=None):
    lags = x[:, 1:] - x[:, :-1]
    if p is not None:
        lags = np.c_[lags, x[:, 0] - x[:, -1] + p]
    return lags
```

While the training code has been optimized for reproducebility rather than flexibility, you can use our training code 
``train.py`` and data structure ``light_curve.py`` as a starter code. 
The data structure has orginally been adapted from 
[Naul et al. 2018](https://github.com/bnaul/IrregularTimeSeriesAutoencoderPaper). To train a model, create a 
LightCurve() object for each of your light curves, and save the list of LightCurve() objects to a pickle file. From
here, follow instructions for reproduction.