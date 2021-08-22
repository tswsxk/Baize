<p align="center">
  <img width="500" src="docs/_static/Baize.png">
</p>

# Baize

[![VERSION](https://img.shields.io/pypi/pyversions/PyBaize)](https://pypi.python.org/pypi/PyBaize)
[![PyPI](https://img.shields.io/pypi/v/PyBaize.svg)](https://pypi.python.org/pypi/PyBaize)
[![test](https://github.com/tswsxk/Baize/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/tswsxk/Baize/actions/workflows/python-test.yml)
[![codecov](https://codecov.io/gh/tswsxk/Baize/branch/main/graph/badge.svg?token=O0MG80VK6U)](https://codecov.io/gh/tswsxk/Baize)

Beyond Artifitial Intelligence modelZoo and boilerplatE (BAIZE) is designed for helping researchers to quickly 
build a deep-learning-based project.

Organizing your code with BAIZE makes your code:

1. Keep architecture consistent by providing both framework and boilerplate 
2. More readable by decoupling the research code from the engineering
2. Keep all the flexibility in framework, which removes a ton of boilerplate; and keep enough details in boilerplate, which give more space for engineers
3. Easier to reproduce and have enough details 
4. Less error-prone by automating most of the training and evaluation, including model persistence and progress monitoring 
5. Combined with lots of deep learning tools, such as nni, for easier and faster tuning
6. Provide many popular reusable modules


## Installation

Install from pypi:
```
pip install PyBaize
```

## Main Features

Baize mainly includes four parts:
* basic project toolkits, like logging
* data process
* deep learning framework:
    1. torch
    2. mxnet
* deep learning boilerplate:
    1. torch
    2. mxnet
* model zoo

### Deep Learning Framework


Using BAIZE light module, we will be able to train and evaluate a network in 5 minutes with friendly log information to both screen and local file.

Before we use the light module, we need to
1. prepare the training dataset
2. define network
3. define the parameters we are going to use in a configuration, such as the model_dir to store the training log and the hyper parameters of network.
4. define how the model will be trained including:
    * the loss function
    * the fit procedure.
    * the trainer (trainer in mxnet, optim in pytorch)
5. we can additionally specify
    * initialization method
    * the evaluation including the data iterator and evaluation procedure
    * hyper parameters searching with nni
    * log to tensorboard
    * command line interface

For convenience, we call the five stages mentioned above as:
1. data preparation
2. network definition
3. configuration
4. training
5. additional

examples: [[torch]](examples/lm_torch.ipynb), [[mxnet]](TBA)
