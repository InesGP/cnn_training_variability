# Uncertain but Useful: Leveraging CNN Variability into Data Augmentation

## Overview

This repository documents the study of numerical variability in CNN training, with a focus on the FastSurfer whole-brain segmentation U-Net. We investigate how numerical uncertainty introduced during training—through Monte Carlo Arithmetic (MCA), random seed perturbations, and weight initialization—can be systematically characterized and leveraged as a data augmentation strategy.

**Key contributions:**
- First evaluation of numerical uncertainty during DL model training for neuroimaging, demonstrating that FastSurfer exhibits higher numerical uncertainty than traditional methods like FreeSurfer
- Demonstration that numerical ensembles derived from these perturbations serve as effective data augmentation without requiring additional data collection
- Application to brain age regression, showing consistent improvements across multiple regression models (Random Forest, SVM, Gradient Boosting)

## MNIST Use Case

Baseline experiments validating the approach on MNIST:

* `mnist_train.py` and `run_mnist_train.sh` contain the code to run MNIST variability experiments
* The following code must be run prior to launching the scripts to download the data:
```python
from torchvision import datasets
dataset1 = datasets.MNIST(f'./data', train=True, download=True)
dataset2 = datasets.MNIST(f'./data', train=False, download=True)
```

## FastSurfer Use Case

Main experiments studying numerical variability during FastSurfer training and data augmentation:

### Data Preparation
* We clone the [FastSurfer code base](https://github.com/Deep-MI/FastSurfer) v2.4.0
* We preprocess the HCP, ABIDE-I, ABIDE-II, ADNI, IXI, LA5C, MIRIAD, OASIS1, OASIS2, and MICA datasets according to FastSurfer authors' methodology using Freesurfer v7.3.2
* All segmentations were quality-controlled by visual inspection in sagittal, coronal, and axial views

### Running Experiments

**Monte Carlo Arithmetic (MCA) perturbations:**
* `launcher_fs.sh` and `run_fuzzy_fast.sh` launch MCA experiments for FastSurfer
* These scripts do not require modification of the FastSurfer code but do require Fuzzy PyTorch and to be launched on a Slurm HPC

**Random seed and weight initialization perturbations:**
* `coronal|axial|sagittal_train_model.sh`, `fs_train_rndseed_model.sh`, and `fs_train_weight_model.sh` launch these experiments
* These require replacing `FastSurfer/FastSurferCNN/train.py` with the local version `fs_train.py`

### Downstream Tasks: Brain Age Prediction

* Brain age regression using ROI volumes extracted from FastSurfer segmentations
* Comparison of multiple data augmentation strategies:
  - **Numerical ensembling**: Multiple FastSurfer iterations with different random seeds
  - **Gaussian noise**: Random perturbations added to ROI features
  - **Synthetic augmentation**: Gaussian Copula and VAE-based data generation
  - **Combined strategies**: Ensemble + synthetic augmentation


## Infrastructure & Containerization
* While the random seed and weight initialization experiments are run in virtual environments on the same nodes, the MCA experiments are run in a Docker/apptainer container located at [inesgp/fuzzy_mnist_sr:train](https://hub.docker.com/r/inesgp/fuzzy_mnist_sr) or [inesgp/fuzzy_sr_fastsurfer:train](https://hub.docker.com/r/inesgp/fuzzy_sr_fastsurfer) depending on the use case
The MCA experiments require specialized tools and are computationally intensive:

* MCA experiments run in Docker/Apptainer containers:
  - [inesgp/fuzzy_mnist_sr:train](https://hub.docker.com/r/inesgp/fuzzy_mnist_sr) for MNIST
  - [inesgp/fuzzy_sr_fastsurfer:train](https://hub.docker.com/r/inesgp/fuzzy_sr_fastsurfer) for FastSurfer

* If your architecture is not compatible with amd64, rebuild the containers:
  1. Build the [Dockerfile from the Fuzzy repository](https://github.com/verificarlo/fuzzy/blob/master/docker/apps/Dockerfile.pytorch) as a base image with instrumented libraries
  2. Use the base image directly for MNIST
  3. Transfer instrumented libraries to a FastSurfer image using [`Dockerfile_fuzzy_fastsurfer`](https://github.com/InesGP/cnn_training_variability/blob/main/Dockerfile_fuzzy_fastsurfer)

### Computational Overhead

MCA adds significant runtime overhead:
* MNIST training: ~30 minutes normally, ~3 days with MCA
* FastSurfer training: correspondingly slower (10-144× slowdown depending on precision)
* MCA can only run on CPU due to instrumentation requirements
