#!/bin/bash
#SBATCH --job-name=weight_init_mnist_train
#SBATCH --mem-per-cpu=10G
#SBATCH --ntasks=1
#SBATCH --time=UNLIMITED
#SBATCH --array=1-11
#SBATCH --output=slurm/%x.out

export OMP_NUM_THREADS=1
export NUMPEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TASK_ID=$SLURM_ARRAY_TASK_ID

# # Fuzzy Training ~ takes approx 3 days
# time apptainer exec --env TASK_ID=$SLURM_ARRAY_TASK_ID -B /home/inesgp/mnist:/mnist -B /home/inesgp/cnn_training:/training /mnt/lustre/inesgp/fuzzy_mnist_sr_grad.sif python3 /mnist/mnist_train.py --save-model 


# Fuzzy Parallel Training -- UNUSED
# time apptainer exec -B ../mnist:/mnist -B /home/inesgp/cnn_training:/training /mnt/lustre/inesgp/fuzzy_fastsurfer_sr.sif python3 /mnist/mnist_train.py --save-model 
# parallel "{} > slurm/fuzzy_sr_seed_{#}_${SLURM_ARRAY_TASK_ID}.log 2>&1" :::: /home/inesgp/cnn_training/fuzzy_seed_iter.txt

# Verrou Training
# time apptainer exec --env TASK_ID=$SLURM_ARRAY_TASK_ID -B ../mnist/:/mnist -B /home/inesgp/cnn_training:/training /mnt/lustre/inesgp/verrou_mnist.sif valgrind --tool=verrou --rounding-mode=average -s --check-nan=no python3 /mnist/mnist_train.py --save-model
  

# IEEE Training ~ 30 mins
# source /home/inesgp/torch_env/bin/activate && time python mnist_train.py --save-model

# # Random seed Training ~ 30 mins
# source /home/inesgp/torch_env/bin/activate && time python mnist_train.py --save-model --seed $SLURM_ARRAY_TASK_ID


# # Wandb Random seed Training ~ 30 mins -- UNUSED
# source /home/inesgp/torch_env/bin/activate && \
# export LOG_DIR='/home/inesgp/cnn_training/wandb' && export SLURM_ARRAY_TASK_ID=1 \
# time python mnist_train_wandb.py --seed $SLURM_ARRAY_TASK_ID


# Weight initialization Training ~ 30 mins
source /home/inesgp/torch_env/bin/activate
init_vals=(
'normal' 
'ones' 
'zeros' 
'identity' 
'dirac' 
'xavier_uniform' 
'xavier_normal' 
'kaiming_uniform' 
'kaiming_normal' 
'orthogonal' 
'sparse'
'uniform'
)

export LOG_DIR='/home/inesgp/cnn_training/weight_init'
export INIT_MODE=${init_vals[(${SLURM_ARRAY_TASK_ID})]}
echo $INIT_MODE
time python mnist_train.py --init-mode ${init_vals[(${SLURM_ARRAY_TASK_ID})]} --save-model


# # Optimizer Training ~ 30 mins
# source /home/inesgp/torch_env/bin/activate
# optim_vals=(
# 'Adagrad'
# 'Adam'
# 'SGD'
# 'LBFGS'
# 'RMSprop'
# )

# export LOG_DIR='/home/inesgp/cnn_training/optim_losses'
# export INIT_MODE=${optim_vals[(${SLURM_ARRAY_TASK_ID})]}
# echo $INIT_MODE
# time python mnist_train.py --optimizer-mode ${optim_vals[(${SLURM_ARRAY_TASK_ID})]}
