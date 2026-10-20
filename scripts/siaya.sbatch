#!/bin/bash
#SBATCH --job-name=siaya-embed
#SBATCH --partition=serc
#SBATCH --constraint=GPU_GEN:AMP|GPU_GEN:HPR
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
: "${DINOSATS_WEIGHTS:?Set the directory containing the model weights}"
: "${DINOSATS_OUT:?Set a new output directory on SCRATCH or GROUP_SCRATCH}"

hostname
printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

mkdir -p "$DINOSATS_OUT"
image=$(realpath "${DINOSATS_IMAGE:-$GROUP_HOME/containers/dinosats.sif}")
weights=$(realpath "$DINOSATS_WEIGHTS")
out=$(realpath "$DINOSATS_OUT")

# Preserve the scheduler's GPU selection across --cleanenv.
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}"
export APPTAINERENV_OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export APPTAINERENV_TMPDIR="$L_SCRATCH_JOB"

apptainer exec --cleanenv --nv "$image" \
    python /opt/dinosats/scripts/check_runtime.py --require-gpu
# The image's model loader expects this fixed weights directory.
apptainer run --cleanenv --nv --bind "$weights:/opt/dinosats/weights:ro" "$image" \
    "$@" --out "$out" --device cuda:0
