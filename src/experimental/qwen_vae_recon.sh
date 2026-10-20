#!/bin/bash
#PBS -P gch51615
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -j oe
# Reconstruct one random anomalous RealIAD image + GT mask per class through the Qwen-Image-2.1 VAE.
# Results go to vis/vae_recon/ (per-class panels + summary.csv).
#
# Usage:
#   qsub src/experimental/qwen_vae_recon.sh
#   qsub -v SEED=1,DTYPE=fp32 src/experimental/qwen_vae_recon.sh
set -euo pipefail

ROOT=${PBS_O_WORKDIR:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$ROOT"
source gerbera/bin/activate

python src/experimental/qwen_vae_recon.py \
    --seed "${SEED:-0}" --dtype "${DTYPE:-bf16}" --out "${OUT:-vis/vae_recon}" ${EXTRA:-}
