#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=henstonny@gmail.com
#SBATCH --job-name=imagebined_vision_perturb
#SBATCH --output=/work/hans1010/slurm_log/%x_%A_%a.out
#SBATCH --error=/work/hans1010/slurm_log/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-19
#SBATCH --account="MST114289"
#SBATCH --partition=gp2d

# original: #SBATCH --array=0-999%20

# ---- job packing ----
TOTAL_EXPERIMENTS=1000
NUM_JOBS=20
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

    export SLURM_ARRAY_TASK_ID=$i

    python zero_shot_test_slurm.py

    echo "====> Finished Sub-experiment Real ID: $i"
done