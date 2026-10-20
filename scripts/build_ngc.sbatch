#!/bin/bash
#SBATCH --job-name=dinosats-build
#SBATCH --partition=serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
hostname
printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
if [[ "${DINOSATS_GPU_TEST:-0}" == 1 ]]; then
    : "${CUDA_VISIBLE_DEVICES:?GPU testing requires a GPU allocation}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
fi
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer/cache"
export APPTAINER_TMPDIR="$L_SCRATCH/apptainer-tmp-$SLURM_JOB_ID"
image="${DINOSATS_IMAGE:-$GROUP_HOME/$USER/dinosats-ngc/dinosats-26.02.sif}"
staged="$image.new-$SLURM_JOB_ID"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$image")"
printf 'image: %s\ntmpdir: %s\ncache: %s\n' "$image" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
df -h "$L_SCRATCH"

# Match remote_outcomes: build on local NVMe and publish only after checks.
apptainer build "$staged" containers/ngc.def
apptainer exec --cleanenv "$staged" python /opt/dinosats/scripts/check_runtime.py

# Optionally validate the final image in the same GPU allocation.
if [[ "${DINOSATS_GPU_TEST:-0}" == 1 ]]; then
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    export APPTAINERENV_OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    apptainer exec --cleanenv --nv "$staged" \
        python /opt/dinosats/scripts/check_runtime.py --benchmark
fi

mv -f "$staged" "$image"
sha256sum "$image"
ls -lh "$image"
