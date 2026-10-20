#!/bin/bash
#PBS -P gch51615
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=6:00:00
#PBS -j oe
# LoRA fine-tuning of Qwen-Image-2.1 (img2img) to map a RealIAD anomalous image to its GT mask, using the
# unmodified diffusers example script on 8 GPUs. Pairs come from build_realiad_mask_pairs.py.
#
# Usage:
#   qsub -o logs/qwen_mask_lora/ src/experimental/qwen_mask_lora.sh
#   qsub -v DATA=outputs/realiad_mask_pairs/<name>,OUT=weights/qwen_mask_lora/<name>,STEPS=10000 ...
set -euo pipefail

ROOT=${PBS_O_WORKDIR:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$ROOT"
source gerbera/bin/activate

DATA=${DATA:-outputs/realiad_mask_pairs/pilot50}
OUT=${OUT:-weights/qwen_mask_lora/pilot50}
VAL_IMAGE=${VAL_IMAGE:-$DATA/$(head -1 "$DATA/holdout.jsonl" | python -c "import json,sys;print(json.load(sys.stdin)['image'])")}
mkdir -p "$OUT"

accelerate launch --multi_gpu --num_processes "${NGPU:-8}" --num_machines 1 --mixed_precision bf16 \
    src/experimental/train_dreambooth_lora_qwenimage21_img2img.py \
    --pretrained_model_name_or_path /groups/gch51615/shared/image/Qwen-Image-2.1 \
    --dataset_name "$DATA" \
    --cond_image_column cond_image --image_column image --caption_column caption \
    --instance_prompt "$(cat "$DATA/prompt.txt")" \
    --output_dir "$OUT" \
    --mixed_precision bf16 \
    --resolution 1024 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --cache_latents \
    --rank "${RANK:-16}" --lora_alpha "${RANK:-16}" \
    --learning_rate "${LR:-1e-4}" \
    --lr_scheduler constant --lr_warmup_steps 0 \
    --max_train_steps "${STEPS:-2000}" \
    --checkpointing_steps 500 \
    --validation_prompt "$(cat "$DATA/prompt.txt")" \
    --validation_image "$VAL_IMAGE" \
    --validation_epochs "${VAL_EPOCHS:-2}" \
    --report_to tensorboard \
    --seed "${SEED:-0}" ${EXTRA:-}
