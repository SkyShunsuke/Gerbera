"""Evaluate a Qwen-Image-2.1 mask LoRA on the held-out RealIAD views written by build_realiad_mask_pairs.py.

For every held-out view, `--samples` masks are generated (one seed each) with the same settings as the training
script's validation (40 steps, true_cfg_scale=1.0). Each sample is binarized at 127 and scored against the GT; the
samples are also averaged into a soft mask (fraction of samples marking a pixel), used for pixel AUROC / AP and
for IoU at a majority vote (>= half the samples) and at the union (any sample).

Normal held-out views (`mask: null` in holdout.jsonl, see build_realiad_mask_manifest.py) have an empty GT: their
segmentation metrics are NaN, and they are scored by `empty_rate` (fraction of samples with no white pixel). The
predicted area fraction (`score`) is used as an image-level anomaly score, giving an image AUROC over all views.

Outputs in --out:
    panels/<cls>__<view>.png   original | GT mask | prediction (mean of samples), side by side
    samples/<cls>__<view>.png  the individual binary samples, side by side
    soft/<cls>__<view>.png     the soft mask at full resolution (value = fraction of samples * 255)
    per_view.csv, per_class.csv, summary.json

Example:
    python src/experimental/eval_qwen_mask_lora.py --lora weights/qwen_mask_lora/pilot50 \
        --data outputs/realiad_mask_pairs/pilot50 --out vis/mask_lora_pilot/holdout
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, roc_auc_score

MODEL_ROOT = "/groups/gch51615/shared/image"
KEYS = ["iou_samples", "iou_best", "hit_rate", "iou_majority", "dice_majority", "iou_union", "pixel_auroc", "pixel_ap"]


def iou_dice(pred, gt):
    inter, union = (pred & gt).sum(), (pred | gt).sum()
    return inter / max(union, 1), 2 * inter / max(pred.sum() + gt.sum(), 1)


def labelled(arr, text, size):
    tile = Image.fromarray(arr).convert("RGB").resize((size, size), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size + 28), "white")
    canvas.paste(tile, (0, 28))
    ImageDraw.Draw(canvas).text((6, 7), text, fill="black")
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default="weights/qwen_mask_lora/pilot50", help="dir or file; 'none' for base model")
    parser.add_argument("--data", default="outputs/realiad_mask_pairs/pilot50")
    parser.add_argument("--model", default="Qwen-Image-2.1", help="name under MODEL_ROOT, or a local path")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--panel_size", type=int, default=512)
    parser.add_argument("--out", default="vis/mask_lora_pilot/holdout")
    args = parser.parse_args()

    data, out = Path(args.data), Path(args.out)
    (out / "panels").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    (out / "soft").mkdir(exist_ok=True)
    prompt = (data / "prompt.txt").read_text().strip()
    views = [json.loads(line) for line in open(data / "holdout.jsonl")]

    model = args.model if Path(args.model).exists() else f"{MODEL_ROOT}/{args.model}"
    pipe = QwenImage21Pipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
    if args.lora != "none":
        pipe.load_lora_weights(args.lora)
    pipe.set_progress_bar_config(disable=True)

    rows = []
    for v in views:
        img = Image.open(data / v["image"]).convert("RGB")
        if v["mask"] is None:
            gt = np.zeros((args.res, args.res), bool)
        else:
            gt = np.array(Image.open(data / v["mask"]).convert("L").resize((args.res, args.res), Image.NEAREST)) > 0
        generators = [torch.Generator("cuda").manual_seed(args.seed + k) for k in range(args.samples)]
        outs = pipe(
            prompt=prompt, image=img.resize((args.res, args.res), Image.BICUBIC), true_cfg_scale=1.0,
            height=args.res, width=args.res, output_resolution=args.res, num_inference_steps=args.steps,
            num_images_per_prompt=args.samples, generator=generators,
        ).images
        gray = np.stack([np.array(o.convert("RGB").convert("L"), dtype=np.float32) / 255 for o in outs])
        binary = gray > 0.5
        soft = binary.mean(0)

        row = dict(cls=v["cls"], view=Path(v["image"]).stem, gt_px=int(gt.sum()), **{k: float("nan") for k in KEYS})
        if gt.any():
            per_sample = [iou_dice(b, gt) for b in binary]
            mean_iou, mean_dice = iou_dice(soft >= 0.5, gt)
            union_iou, _ = iou_dice(soft > 0, gt)
            row.update(
                iou_samples=round(float(np.mean([s[0] for s in per_sample])), 4),
                iou_best=round(float(max(s[0] for s in per_sample)), 4),
                hit_rate=round(float(np.mean([s[0] > 0.1 for s in per_sample])), 3),
                iou_majority=round(float(mean_iou), 4), dice_majority=round(float(mean_dice), 4),
                iou_union=round(float(union_iou), 4),
                pixel_auroc=round(float(roc_auc_score(gt.ravel(), soft.ravel())), 4),
                pixel_ap=round(float(average_precision_score(gt.ravel(), soft.ravel())), 4),
            )
        row.update(empty_rate=round(float(np.mean([not b.any() for b in binary])), 3),
                   score=round(float(soft.mean()), 6), pred_px_majority=int((soft >= 0.5).sum()), path=v["path"])
        rows.append(row)
        print({k: row[k] for k in row if k != "path"}, flush=True)

        s = args.panel_size
        panel = [
            labelled(np.array(img.resize((args.res, args.res))), f"{v['cls']}  original", s),
            labelled((gt * 255).astype(np.uint8), f"GT  {row['gt_px']} px" if gt.any() else "GT  normal (empty)", s),
            labelled((soft * 255).astype(np.uint8),
                     f"prediction (mean of {args.samples})  majority IoU {row['iou_majority']:.2f}  AP {row['pixel_ap']:.2f}"
                     if gt.any() else f"prediction (mean of {args.samples})  empty in {row['empty_rate']:.0%}", s),
        ]
        canvas = Image.new("RGB", (s * len(panel), panel[0].height), "white")
        for i, p in enumerate(panel):
            canvas.paste(p, (i * s, 0))
        canvas.save(out / "panels" / f"{row['view']}.png")
        Image.fromarray((soft * 255).round().astype(np.uint8)).save(out / "soft" / f"{row['view']}.png")
        sample_strip = np.concatenate([(b * 255).astype(np.uint8) for b in binary], axis=1)
        Image.fromarray(sample_strip).resize((s * args.samples, s), Image.NEAREST).save(out / "samples" / f"{row['view']}.png")

    keys = KEYS + ["empty_rate"]
    mean = lambda rs, k: round(float(np.nanmean([r[k] for r in rs])), 4) if any(r[k] == r[k] for r in rs) else ""
    with open(out / "per_view.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["cls"]].append(r)
    with open(out / "per_class.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cls", "n"] + keys)
        for cls, rs in by_cls.items():
            writer.writerow([cls, len(rs)] + [mean(rs, k) for k in keys])
        writer.writerow(["ALL", len(rows)] + [mean(rows, k) for k in keys])

    anom = [r for r in rows if r["gt_px"] > 0]
    normal = [r for r in rows if r["gt_px"] == 0]
    summary = {"n_anomalous": len(anom), "n_normal": len(normal)}
    summary.update({f"anomalous_{k}": mean(anom, k) for k in keys} if anom else {})
    if normal:
        summary["normal_empty_rate"] = mean(normal, "empty_rate")
        summary["normal_empty_majority"] = round(float(np.mean([r["pred_px_majority"] == 0 for r in normal])), 4)
    if anom and normal:
        labels, scores = [r["gt_px"] > 0 for r in rows], [r["score"] for r in rows]
        summary["image_auroc"] = round(float(roc_auc_score(labels, scores)), 4)
        summary["image_ap"] = round(float(average_precision_score(labels, scores)), 4)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
