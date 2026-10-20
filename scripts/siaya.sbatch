#!/bin/bash
#SBATCH --job-name=siaya-embed
#SBATCH --partition=serc
#SBATCH --constraint=GPU_GEN:AMP|GPU_GEN:HPR
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
: "${DINOSATS_OUT:?Set a new output directory on SCRATCH or GROUP_SCRATCH}"

hostname
printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

mkdir -p "$DINOSATS_OUT"
image=$(realpath "${DINOSATS_IMAGE:-$GROUP_HOME/containers/dinosats.sif}")
out=$(realpath "$DINOSATS_OUT")
repo="${SLURM_SUBMIT_DIR:?Submit from the DinoSats repository root}"
test -f "$repo/scripts/build_siaya_full.py"
env_args=()
if [[ -f "$repo/.env" ]]; then
    env_args=(--env-file "$repo/.env")
fi

# Preserve the scheduler's GPU selection across --cleanenv.
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}"
export APPTAINERENV_OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export APPTAINERENV_TMPDIR="$L_SCRATCH_JOB"
export APPTAINERENV_HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH/huggingface/hub}"
export APPTAINERENV_HF_XET_CACHE="${HF_XET_CACHE:-$SCRATCH/huggingface/xet}"
# Keep saved-login locations and optional token/offline settings across --cleanenv.
for name in HF_HOME HF_TOKEN_PATH HF_TOKEN HF_HUB_OFFLINE; do
    if [[ -v "$name" ]]; then
        export "APPTAINERENV_$name=${!name}"
    fi
done

apptainer exec --cleanenv --nv "$image" \
    python /opt/dinosats/scripts/check_runtime.py --require-gpu
# Run the checked-out code so loader updates do not require rebuilding the image.
apptainer exec --cleanenv --nv "$image" \
    uv run --no-project --no-cache --python /opt/venv/bin/python "${env_args[@]}" \
    python -u "$repo/scripts/build_siaya_full.py" \
    --inventory "${DINOSATS_INVENTORY:-$SCRATCH/inventory.json}" \
    --requests "$SCRATCH/vantor_request.csv" \
    --clouds "$SCRATCH/vantor_browse" \
    "$@" --out "$out" --device cuda:0
