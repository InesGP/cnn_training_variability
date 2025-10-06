#!/bin/bash
# Usage: ./launch_orientation.sh coronal|axial|sagittal

ORIENT=$1

if [[ -z "$ORIENT" ]]; then
  echo "Usage: $0 {coronal|axial|sagittal}"
  exit 1
fi

if [[ "$ORIENT" != "coronal" && "$ORIENT" != "axial" && "$ORIENT" != "sagittal" ]]; then
  echo "Error: orientation must be one of coronal, axial, sagittal"
  exit 1
fi

# Submit to SLURM with a dynamic job name and orientation argument
sbatch --job-name=fuzzy_fast_${ORIENT} run_fuzzy_fast.sh "$ORIENT"

