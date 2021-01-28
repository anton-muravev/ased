# ASED (Architecture Search by Estimating Distributions)
This code accompanies the following paper:

> [A. Muravev, J. Raitoharju and M. Gabbouj, "Neural Architecture Search by Estimation of Network Structure Distributions,"](https://ieeexplore.ieee.org/document/9328761) in IEEE Access, vol. 9, pp. 15304-15319, 2021, doi: 10.1109/ACCESS.2021.3052996.


# Requirements
Python >=3.6, pytorch ==1.4, torchvision==0.5, scikit-learn

# Implementation
The repository contains the following code files:

* `ased.py` - core functionality of the ASED algorithm
* `ased_util.py` - non-ASED specific utility functions
* `astore.py` - `AStore` class, wrapping around the Python dictionary with disk storage operations
* `utorch.py` - generic utility functions for deep learning
* `/scripts/` - scripts to run ASED on CIFAR10 and CIFAR100 (with fixed random seed)
  * `/scripts/ased_initialize_cifar*.py` - create and store the initialization point for ASED
  * `/scripts/ased_search_cifar*.py` - run the baseline ASED and store the results in a file
  * `/scripts/ased_search_bounded_cifar*.py` - run the ASED variant with probability bounding and store the results in a file
  * `/scripts/ased_search_inversion1_cifar*.py` - run the ASED variant with full prototype inversion and store the results in a file
  * `/scripts/ased_search_inversion2_cifar*.py` - run the ASED variant with partial prototype inversion and store the results in a file
  * `/scripts/ased_validate_*.py` - run the evaluation of the ASED output on the validation set

The code files in the root folder are internally documented and commented. The provided scripts represent one possible example of use on the multi-GPU setup and can be generalized for other datasets or GPU allocation patterns.

## Input data
CIFAR10/100 datasets have to be downloaded separately (e.g. using the functionality of CIFAR wrappers from `torchvision.datasets`). Paths to the dataset folders must be provided as arguments to the scripts (see below).

Initialization and search scripts both produce output files that are required as inputs to the search and validation scripts respectively.

## Command line arguments

By default the numeric arguments are set to the values used in the paper. All of the arguments referencing input and output file names are mandatory.

Initialization arguments:
```
--cifarpath      # path to the dataset folder
--out            # name/path of the output file
--nlayer         # number of layers in the initial prototype
--gpus           # number of GPU devices to use
--netcount       # number of networks to sample per GPU
--workers        # number of CPU workers per GPU
```
Search arguments:
```
--cifarpath      # path to the dataset folder
--init           # full path to the initialization file
--out            # full path to the output file
--iter           # number of ASED search iterations to run
--dense          # if given a value, enables semi-dense connection pattern with the given value
--residual       # if given a value, enables residual connection pattern with the given value
--bound          # enables probability bounding with the given threshold
--invth          # prototype norm threshold for inversion (inversion variant only)
--protolimit     # inversion count limit before search is terminated (inversion variant only)
--gpus           # number of GPU devices to use
--netcount       # number of networks to sample per GPU
--workers        # number of CPU workers per GPU
--resume         # if given a value, resumes the existing search from the given iteration
```
Validation arguments:
```
--cifarpath      # path to the dataset folder
--input          # full path to the search file
--epochs         # number of epochs to train for
--channels       # number of channels to use for all the layers
--dense          # if given a value, uses semi-dense connection pattern with the given value
--residual       # if given a value, uses residual connection pattern with the given value
--nextgen        # if present, the structure is inferred from the next prototype instead of the current one
--workers        # number of CPU workers
```

# Acknowledgement
This work was supported by the European Union’s Horizon 2020 Research and Innovation Programme under Grant 871449 (OpenDR).

# Citation

```
@article{muravev2021ASED,
  author={A. {Muravev} and J. {Raitoharju} and M. {Gabbouj}},
  journal={IEEE Access},
  title={Neural Architecture Search by Estimation of Network Structure Distributions},
  year={2021},
  volume={9},
  pages={15304-15319},
  doi={10.1109/ACCESS.2021.3052996}}
```
