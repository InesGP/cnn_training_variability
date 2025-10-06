#!/encs/bin/bash

for i in {1..10}; do
    sbatch ./scripts/train/sagittal_train_model.sh $i
    sbatch ./scripts/train/axial_train_model.sh $i
    sbatch ./scripts/train/coronal_train_model.sh $i
done

