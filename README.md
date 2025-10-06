# cnn_training_variability

* Currently work in progress, pairwise min Dice score figures and brain age predictions for random forest and SVM models using the random forest FastSurfer ensemble can be re-generated in the `Dice Scores` and `ROI Volumes` sections of the FastSurfer notebook respectively
* While the random seed and weight initialization experiments are run in virtual environments on the same nodes, the MCA experiments are run in a Docker/apptainer container located at [inesgp/fuzzy_mnist_sr:train](https://hub.docker.com/repository/docker/inesgp/fuzzy_mnist_sr/general) or [inesgp/fuzzy_sr_fastsurfer:train](https://hub.docker.com/repository/docker/inesgp/fuzzy_sr_fastsurfer/general) depending on the use case

### MNIST Use case
* `mnist_train.py` and  `run_mnist_train.sh` contain the code to run the MNIST variability experiments
* The following code must be run previous to launching the scripts as they assume the data has already been downloaded
```python
from torchvision import datasets
dataset1 = datasets.MNIST(f'./data', train=True, download=False, )
dataset2 = datasets.MNIST(f'./data', train=False,)
```

### FastSurfer Use case
* We first clone the [FastSurfer code base](https://github.com/Deep-MI/FastSurfer) v2.4.0
* We preprocess the HCP, ABIDE-I, ABIDE-II, ADNI, IXI, LA5C, MIRIAD, OASIS1, OASIS2 and MICA datasets according to the FastSurfer authors' methodology
* `launcher_fs.sh` and `run_fuzzy_fast.sh` contains the code to launch the MCA experiment for FastSurfer and does not require any modification of the FastSurfer code
* `coronal|axial|sagittal_train_model.sh`, `fs_train_rndseed_model.sh` and `fs_train_weight_model.sh` launch the weight and random seed experiments but require `FastSurfer/FastSurferCNN/train.py` to be overwritten with the local version `fs_train.py`
