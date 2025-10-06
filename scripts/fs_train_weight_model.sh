#!/encs/bin/bash
init_mode=("uniform" "normal" "ones" "zeros" "identity" "eye" "dirac" "xavier_uniform" "xavier_normal" "kaiming_uniform" "kaiming_normal" "orthogonal" "sparse")
for i in "${init_mode[@]}"; do
    echo $i
    sbatch ./scripts/train/sagittal_train_model.sh 1 $i
    sbatch ./scripts/train/axial_train_model.sh 1 $i
    sbatch ./scripts/train/coronal_train_model.sh 1 $i
done

