#!/bin/bash
#PBS -P gch51615
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=4:00:00
#PBS -j oe
# Generate masks for the held-out RealIAD views with a trained mask LoRA and score them.
#
# Usage:
#   qsub -o logs/qwen_mask_lora/ src/experimental/eval_qwen_mask_lora.sh
#   qsub -v LORA=none,OUT=vis/mask_lora_pilot/holdout_base ...   # base model, no LoRA
#   full run:  qsub -v LORA=weights/qwen_mask_lora/full,DATA=outputs/realiad_mask_pairs/full,OUT=vis/mask_lora_full/holdout ...
set -euo pipefail

ROOT=${PBS_O_WORKDIR:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$ROOT"
source gerbera/bin/activate

python src/experimental/eval_qwen_mask_lora.py \
    --lora "${LORA:-weights/qwen_mask_lora/pilot50}" \
    --data "${DATA:-outputs/realiad_mask_pairs/pilot50}" \
    --out "${OUT:-vis/mask_lora_pilot/holdout}" \
    --model "${MODEL:-Qwen-Image-2.1}" --samples "${SAMPLES:-4}" --seed "${SEED:-0}" ${EXTRA:-}
