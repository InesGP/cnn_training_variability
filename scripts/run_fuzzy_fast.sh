#!/bin/bash
#SBATCH --time=168:0:0
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --array=1-10
#SBATCH --output=slurm/%x_%a.out
#SBATCH --error=slurm/%x_%a.err

module load apptainer/1.2

ORIENT=$1

if [[ -z "$ORIENT" ]]; then
  echo "Error: Orientation argument missing."
  exit 1
fi

apptainer exec \
-B /scratch/vinuyans/fastsurfer-dataset/hdf5_sets/:/data \
-B /scratch/ine5/:/val_data \
-B /scratch/ine5/FastSurfer/experiments_${SLURM_ARRAY_TASK_ID}_${ORIENT}:/output \
-B /scratch/ine5/FastSurfer/FastSurferCNN/run_model.py:/fastsurfer/FastSurferCNN/run_model.py \
-B /scratch/ine5/FastSurfer/FastSurferCNN/train.py:/fastsurfer/FastSurferCNN/train.py \
-B /scratch/ine5/FastSurfer/FastSurferCNN/data_loader/loader.py:/fastsurfer/FastSurferCNN/data_loader/loader.py \
--env SLURM_CPUS_PER_TASK=192 \
/scratch/ine5/fuzzy_fastsurfer_sr_train.sif \
time python3 /fastsurfer/FastSurferCNN/run_model.py \
DATA.PATH_HDF5_TRAIN /data/train_${ORIENT}_dataset.hdf5 \
DATA.PATH_HDF5_VAL /val_data/val_${ORIENT}_dataset.hdf5 \
TRAIN.NUM_EPOCHS 70 LOG_DIR /output TRAIN.RESUME True TRAIN.RESUME_EXPR_NUM FastSurferVINN

