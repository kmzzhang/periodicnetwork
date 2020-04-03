## Improved classification of periodic variable stars with phase-invariant neural networks

Code associated with the paper:
  > [Improved classification of periodic variable stars with phase-invariant neural networks]\
  > Keming Zhang & Joshua Bloom\
  > currently under review

This repository contains pytorch implementations of the cyclic-permutation invariant networks used in our study.
It also contains code to train and test the invariant networks as well as baseline networks.

Neural network implementations can be found under ./model/

**Description of files**:\
**clean.py**: preform outlier rejection on the raw datasets and save to the same directory\
**train.py**: train neural networks on variable star light curve datasets\
**ppmnist.py**: train neural networks on the pp-mnist task


The variable star light curve datasets used in this study have been uploaded to zenodo, and can be downloaded using the
script in ./data/download.sh

Example usage
```
sh ./data/download.sh
python clean.py --file macho_raw.sh
python train.py --filename macho_raw_cleaned.pkl --network iresnet --depth 6 --hidden 32 --max_hidden 32 --path results
```
