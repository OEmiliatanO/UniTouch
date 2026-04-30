#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=henstonny@gmail.com
#SBATCH --job-name=imagebined_vision_perturb
#SBATCH --output=/work/hans1010/slurm_log/%x_%A_%a.out
#SBATCH --error=/work/hans1010/slurm_log/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --account="MST114289"
#SBATCH --partition=gp1d

# ---- job packing ----
TOTAL_EXPERIMENTS=1
NUM_JOBS=1
CHUNK_SIZE=$(( (TOTAL_EXPERIMENTS + NUM_JOBS - 1) / NUM_JOBS ))
START_ID=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_ID=$(( START_ID + CHUNK_SIZE - 1 ))

# environment setup
module purge
module load cuda/12.8
module load miniconda3
source activate /home/hans1010/miniconda3/envs/imagebind

# make sure logs directory exists
mkdir -p /work/hans1010/slurm_log

echo "Pack Job ID: $SLURM_ARRAY_TASK_ID"
echo "Running experiments from ID $START_ID to $END_ID"

for (( i=START_ID; i<=END_ID; i++ )); do
    if [ "$i" -ge "$TOTAL_EXPERIMENTS" ]; then
        break
    fi

    echo "====> Starting Sub-experiment Real ID: $i"

    RANDOM_PORT=$((RANDOM % 1000 + 25000))

    export SLURM_ARRAY_TASK_ID=$i

    torchrun --nproc_per_node=4 --master_port=$RANDOM_PORT train.py \
        --strategy "unitouch" \
        --seed $i \
        --freeze_vision \
        --preserver_imagenet_features \
        --exp_name "" \
        --debug \
        --epochs 10 \
        --batch_size 20 \
        --testing_batch_size 32 \
        --imagenet_testing_batch_size 32 \
        --TWCC

    echo "====> Finished Sub-experiment Real ID: $i"
done
