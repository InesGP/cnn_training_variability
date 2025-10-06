#!/encs/bin/bash
#SBATCH --time=01-12:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

# $1 = RNG_SEED
# $2 = weight init mode

SINGULARITY=/encs/pkg/singularity-3.10.4/root/bin/singularity

mode=$2

srun $SINGULARITY exec --nv --writable-tmpfs --bind /speed-scratch:/speed-scratch --env INIT_MODE=$mode fastsurfer.sif \
python3 $(pwd)/FastSurferCNN/run_model.py --cfg ./FastSurferCNN/config/FastSurferVINN_axial.yaml \
EXPR_NUM FastSurferVINN_axial_$mode \
RNG_SEED $1 \
SUMMARY_PATH FastSurferVINN/summary/FastSurferVINN_axial_$mode \
CONFIG_LOG_PATH FastSurferVINN/config/FastSurferVINN_axial_$mode \
LOG_DIR weight_init_models/FastsurferVINN_axial_$mode

