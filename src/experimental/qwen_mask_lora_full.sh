#!/bin/bash
#PBS -P gch51615
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -j oe
# LoRA fine-tuning of Qwen-Image-2.1 (img2img) on the full RealIAD manifest from build_realiad_mask_manifest.py:
# every anomalous test view (target = GT mask) and every normal train view (target = black mask), minus held-out
# objects. Same hyper-parameters as the pilot (qwen_mask_lora.sh); the data is streamed from disk.
#
# Usage:
#   qsub -o logs/qwen_mask_lora/ src/experimental/qwen_mask_lora_full.sh             # 8 GPUs, 1 epoch
#   smoke test: qsub -q rt_HG -l walltime=1:00:00 -v NGPU=1,STEPS=20,OUT=weights/qwen_mask_lora/smoke ...
#   resume:     qsub -v EXTRA="--resume_from_checkpoint latest" ...
#   elsewhere:  MODEL=<path to Qwen-Image-2.1> NGPU=8 bash src/experimental/qwen_mask_lora_full.sh  (no PBS needed)
set -euo pipefail

ROOT=${PBS_O_WORKDIR:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$ROOT"
source gerbera/bin/activate

DATA=${DATA:-outputs/realiad_mask_pairs/full}
OUT=${OUT:-weights/qwen_mask_lora/full}
VAL_IMAGE=${VAL_IMAGE:-$DATA/$(head -1 "$DATA/holdout.jsonl" | python -c "import json,sys;print(json.load(sys.stdin)['image'])")}
mkdir -p "$OUT"

NGPU=${NGPU:-8}
MULTI_GPU=$([ "$NGPU" -gt 1 ] && echo --multi_gpu || true)
# no STEPS -> one epoch over the manifest
STEP_ARGS=$([ -n "${STEPS:-}" ] && echo "--max_train_steps $STEPS" || echo "--num_train_epochs ${EPOCHS:-1}")
accelerate launch $MULTI_GPU --num_processes "$NGPU" --num_machines 1 --mixed_precision bf16 \
    src/experimental/train_qwen_mask_lora_stream.py \
    --pretrained_model_name_or_path "${MODEL:-/groups/gch51615/shared/image/Qwen-Image-2.1}" \
    --manifest "$DATA/train.jsonl" \
    --instance_prompt "$(cat "$DATA/prompt.txt")" \
    --output_dir "$OUT" \
    --mixed_precision bf16 \
    --resolution 1024 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --dataloader_num_workers 8 \
    --rank "${RANK:-16}" --lora_alpha "${RANK:-16}" \
    --learning_rate "${LR:-1e-4}" \
    --lr_scheduler constant --lr_warmup_steps 0 \
    $STEP_ARGS \
    --checkpointing_steps "${CKPT_STEPS:-1000}" \
    --validation_prompt "$(cat "$DATA/prompt.txt")" \
    --validation_image "$VAL_IMAGE" \
    --validation_epochs 1 \
    --report_to tensorboard \
    --seed "${SEED:-0}" ${EXTRA:-}
