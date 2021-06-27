# Snake Species Identification

## Overview

This repository contains supplementary material to the working note paper
**A Deep Learning Method for Visual Recognition of Snake Species**
submitted to the SnakeCLEF 2021 challenge - a part of LifeCLEF 2021 workshop.

The goal of the challenge was creating a method for image-based snake species identification. 
The proposed method is based on deep residual neural networks, namely 
ResNet, ResNeXt and ResNeSt, fine-tuned from ImageNet-1k pre-trained checkpoints.

## Content

The repository contains Jupyter Notebooks for training and testing CNNs:
* [01a_data_exploration.ipynb](01a_data_exploration.ipynb) - 
  The notebook contains a high-level exploration of the training dataset
  including visualization of snake species distribution
  and geographical distribution of images.
* [01b_image_exploration.ipynb](01b_image_exploration.ipynb) -
  The notebook contains exploration of images in the training dataset including 
  examples of noisy data from Flickr and estimation of the confidence interval 
  of noisy images in Flickr using Student's t-distribution.
* [02_training.ipynb](02_training.ipynb) -
  The notebook contains a training pipeline in `fastai` framework.
* [03a_testing.ipynb](03a_testing.ipynb) -
  The notebook contains a pipeline for creating predictions on the validation and test sets. 
* [03b_ensemble.ipynb](03b_ensemble.ipynb) -
  The notebook contains a pipeline for creating ensemble model predictions by applying
  majority voting strategy. 
* [04_evaluating.ipynb](04_evaluating.ipynb) - 
  The notebook contains a pipeline for evaluating predictions of a single CNN.
  This includes classification report with precision, recall and F1 scores,
  and examples of correct and incorrect predictions with high or low confidence.
* [05_submission_results.ipynb](05_submission_results.ipynb) -
  The notebook contains classification scores of CNNs submitted to the SnakeCLEF 2021 challenge. 

Additionally, the repository contains Jupyter Notebooks with experiments comparing 
prediction scores under various setups:
* [01_models.ipynb](experiments/01_models.ipynb) -
  The notebook compares residual networks ResNet, ResNeXt and ResNeSt with 50 and 101 layers.
* [02_loss_functions.ipynb](experiments/02_loss_functions.ipynb) -
  The notebook compares loss functions Cross Entropy, Weighted Cross Entropy and F1 Loss.
* [03_optimizers.ipynb](experiments/03_optimizers.ipynb) -
  The notebook compares optimization algorithms SGD with momentum and Adam
  with and without one cycle schedule policy.
* [04_mixed_precision.ipynb](experiments/04_mixed_precision.ipynb) -
  The notebook compares the single precision training, used by default, 
  and the mixed precision training.

## Getting Started
### Dataset
The training dataset was provided by the organizers of SnakeCLEF 2021 challenge,
and it is not part of this repository. The dataset was obtained through 
[AIcrowd](https://www.aicrowd.com/challenges/snakeclef2021-snake-species-identification-challenge).

### Package Dependencies
The proposed methods were developed using `Python=3.7` with `PyTorch=1.7` machine learning framework
and a `fastai=2.3` framework build on top of PyTorch.
All models were fine-tuned from pre-trained PyTorch Image Models, `timm=0.4`.
Additionally, the repository requires packages: 
`numpy`, `pandas`, `scikit-learn`, `matplotlib` and `seaborn`.

To install required packages with PyTorch for CPU run:
```bash
pip install -r requirements.txt
```

For PyTorch with GPU run:
```bash
pip install -r requirements_gpu.txt
```

The requirement files do not contain `jupyterlab` nor any other IDE.
To install `jupyterlab` run
```bash
pip install jupyterlab
```

The Jupyter Notebooks use `pycodestyle` magic function
which checks Python code against the
[PEP 8](https://www.python.org/dev/peps/pep-0008/) style conventions
allowing to maintain a well readable and clean code.
To install `pycodestyle` for `jupyterlab` run:
```bash
pip install flake8 pycodestyle_magic
```

## Contact
**Rail Chamidullin** - chamidullinr@gmail.com  - [Github account](https://github.com/chamidullinr)
