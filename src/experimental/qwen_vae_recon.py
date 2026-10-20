"""Check whether the Qwen-Image-2.1 VAE latent preserves tiny RealIAD anomalies and GT-mask semantics.

For each RealIAD class one anomalous test view (and its GT mask) is picked at random, encoded to the
16x-downsampled latent and decoded back. Two things are measured:
  * image: PSNR over the whole image vs. inside the (dilated) defect region, and the residual there
  * mask: the GT mask is fed as a white-on-black RGB image; the decoded image is thresholded and
          compared with the GT (IoU / Dice / recall of the defect pixels)

Example:
    python src/experimental/qwen_vae_recon.py --out vis/vae_recon --seed 0
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage21
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.realiad import RealIAD, REALIAD_CLASSES

MODEL_ROOT = "/groups/gch51615/shared/image"


def to_vae_input(rgb):
    """HxWx3 uint8 -> 1x4x1xHxW in [-1, 1] (opaque RGBA, as the pipeline does)."""
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    x = torch.from_numpy(rgba).permute(2, 0, 1).float() / 127.5 - 1.0
    return x[None, :, None]


@torch.no_grad()
def reconstruct(vae, rgb, dtype):
    x = to_vae_input(rgb).to("cuda", dtype)
    z = vae.encode(x).latent_dist.mode()
    y = vae.decode(z, return_dict=False)[0][0, :3, 0]
    y = ((y.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return y, tuple(z.shape)


def psnr(a, b, region=None):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    mse = d[region].mean() if region is not None else d.mean()
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-10))


def crop_box(mask, pad=48, min_size=128):
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min(), min_size) // 2 + pad
    h, w = mask.shape
    y0, x0 = max(cy - half, 0), max(cx - half, 0)
    return y0, min(cy + half, h), x0, min(cx + half, w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/realiad")
    parser.add_argument("--meta_dir", default="data/realiad_meta/realiad_jsons")
    parser.add_argument("--model", default="Qwen-Image-2.1")
    parser.add_argument("--classes", nargs="+", default=REALIAD_CLASSES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--out", default="vis/vae_recon")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    vae = AutoencoderKLQwenImage21.from_pretrained(f"{MODEL_ROOT}/{args.model}", subfolder="vae", torch_dtype=dtype)
    vae = vae.to("cuda").eval()
    rng = random.Random(args.seed)

    rows = []
    for cls in args.classes:
        ds = RealIAD(args.data_root, cls, 1024, "test", args.meta_dir)
        idx = rng.choice(ds.anom_indices)
        img_path, mask_path = ds.img_files[idx], ds.masks[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        gt = np.array(Image.open(mask_path).convert("L").resize(img.shape[1::-1], Image.NEAREST)) > 0
        # VAE needs sides divisible by 16
        h, w = (img.shape[0] // 16) * 16, (img.shape[1] // 16) * 16
        img, gt = img[:h, :w], gt[:h, :w]

        rec, zshape = reconstruct(vae, img, dtype)
        mask_rgb = np.repeat((gt * 255).astype(np.uint8)[..., None], 3, axis=-1)
        mask_rec, _ = reconstruct(vae, mask_rgb, dtype)
        mask_soft = mask_rec.mean(-1) / 255.0
        mask_bin = mask_soft > 0.5

        # residual statistics in the defect vs. a 16px ring (one latent cell) of normal context
        region = cv2.dilate(gt.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ring = (cv2.dilate(gt.astype(np.uint8), np.ones((33, 33), np.uint8)) > 0) & ~region
        err = np.abs(img.astype(np.float32) - rec.astype(np.float32)).mean(-1)
        inter, union = (mask_bin & gt).sum(), (mask_bin | gt).sum()
        row = dict(
            cls=cls, path=img_path, anom_type=Path(img_path).parts[-3],
            area_px=int(gt.sum()), area_latent_cells=round(gt.sum() / 256, 2), latent=str(zshape),
            psnr_img=round(psnr(img, rec), 2), psnr_defect=round(psnr(img, rec, region), 2),
            mae_defect=round(float(err[region].mean()), 2), mae_ring=round(float(err[ring].mean()), 2),
            mask_iou=round(inter / max(union, 1), 4), mask_dice=round(2 * inter / max(gt.sum() + mask_bin.sum(), 1), 4),
            mask_recall=round(inter / max(gt.sum(), 1), 4), mask_soft_in=round(float(mask_soft[gt].mean()), 3),
            mask_soft_out_max=round(float(mask_soft[~region].max()), 3),
        )
        rows.append(row)
        print(row, flush=True)

        y0, y1, x0, x1 = crop_box(gt)
        crop = lambda a: a[y0:y1, x0:x1]
        contour = lambda a: cv2.drawContours(a.copy(), cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)[0], -1, (255, 0, 0), 2)
        panels = [
            (contour(img), "input (GT contour)"), (rec, f"recon  PSNR {row['psnr_img']}"),
            (crop(img), "input (crop)"), (crop(rec), f"recon (crop)  defect PSNR {row['psnr_defect']}"),
            (crop(err), "|input - recon| (crop)"),
            (crop(gt), f"GT mask (crop)  {row['area_px']} px"), (crop(mask_soft), "decoded mask (crop)"),
            (crop(mask_bin), f"decoded > 0.5  IoU {row['mask_iou']}"),
        ]
        fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
        for ax, (a, t) in zip(axes, panels):
            ax.imshow(a, cmap="inferno" if a is panels[4][0] else "gray", vmin=0,
                      vmax=None if a.ndim == 3 else (64 if a is panels[4][0] else 1))
            ax.set_title(t, fontsize=8)
            ax.axis("off")
        fig.suptitle(f"{cls} / {row['anom_type']} / {Path(img_path).name}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{cls}.png", dpi=110)
        plt.close(fig)

    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = ["psnr_img", "psnr_defect", "mae_defect", "mae_ring", "mask_iou", "mask_dice", "mask_recall"]
    print("mean:", {k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys})


if __name__ == "__main__":
    main()
