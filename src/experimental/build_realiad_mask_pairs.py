"""Build (inspection image -> GT mask) pairs from RealIAD anomalous test views for Qwen-Image-2.1 img2img LoRA.

Writes a parquet dataset readable by `load_dataset(<out>)` with the columns the diffusers
`train_dreambooth_lora_qwenimage21_img2img.py` script expects:
    cond_image  the anomalous inspection image (condition)
    image       the binarized GT mask, white defect on black (target)
    caption     the segmentation instruction
plus `cls` / `path` for bookkeeping.

Per class, `--per_class` anomalous objects are drawn at random and one anomalous view is taken from each, so the
training views come from distinct objects. `--holdout_per_class` further views are taken from objects that were not
drawn for training, and written as image/mask files plus `holdout.jsonl` for validation.

Example:
    python src/experimental/build_realiad_mask_pairs.py --per_class 50 --out outputs/realiad_mask_pairs/pilot50
"""
import argparse
import importlib.util
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Image as ImageFeature, Value
from PIL import Image

# src/datasets would shadow HuggingFace `datasets` on sys.path, so load the RealIAD module by file path
_spec = importlib.util.spec_from_file_location(
    "realiad", Path(__file__).resolve().parents[1] / "datasets" / "realiad.py")
_realiad = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_realiad)
RealIAD, REALIAD_CLASSES = _realiad.RealIAD, _realiad.REALIAD_CLASSES

PROMPT = (
    "Produce the anomaly segmentation mask for this inspection image. Mark defective pixels white and all other "
    "pixels black. Preserve their exact positions. Return an opaque mask."
)


def mask_png(mask_path, size):
    """GT masks are anti-aliased; binarize like RealIAD.__getitem__ (any annotated pixel is defective)."""
    m = np.array(Image.open(mask_path).convert("L").resize(size, Image.NEAREST)) > 0
    buf = io.BytesIO()
    Image.fromarray((m * 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/realiad")
    parser.add_argument("--meta_dir", default="data/realiad_meta/realiad_jsons")
    parser.add_argument("--classes", nargs="+", default=REALIAD_CLASSES)
    parser.add_argument("--per_class", type=int, default=50)
    parser.add_argument("--holdout_per_class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/realiad_mask_pairs/pilot50")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "holdout").mkdir(exist_ok=True)
    rng = random.Random(args.seed)

    rows = defaultdict(list)
    holdout = []
    for cls in args.classes:
        ds = RealIAD(args.data_root, cls, 1024, "test", args.meta_dir)
        # an object is one NG/<defect type>/<sample id> folder holding its 5 views
        by_object = defaultdict(list)
        for i in ds.anom_indices:
            by_object[Path(ds.img_files[i]).parent].append(i)
        objects = sorted(by_object)
        rng.shuffle(objects)
        n_train = min(args.per_class, len(objects) - args.holdout_per_class)
        for obj in objects[:n_train]:
            i = rng.choice(by_object[obj])
            with open(ds.img_files[i], "rb") as f:
                img_bytes = f.read()
            size = Image.open(io.BytesIO(img_bytes)).size
            rows["cond_image"].append({"bytes": img_bytes, "path": None})
            rows["image"].append({"bytes": mask_png(ds.masks[i], size), "path": None})
            rows["caption"].append(PROMPT)
            rows["cls"].append(cls)
            rows["path"].append(ds.img_files[i])
        for obj in objects[n_train:n_train + args.holdout_per_class]:
            i = rng.choice(by_object[obj])
            stem = f"{cls}__{Path(ds.img_files[i]).stem}"
            Image.open(ds.img_files[i]).convert("RGB").save(out / "holdout" / f"{stem}.png")
            size = Image.open(ds.img_files[i]).size
            (out / "holdout" / f"{stem}_mask.png").write_bytes(mask_png(ds.masks[i], size))
            holdout.append(dict(cls=cls, path=ds.img_files[i], image=f"holdout/{stem}.png",
                                mask=f"holdout/{stem}_mask.png"))
        print(f"{cls}: {n_train} train views from {len(objects)} anomalous objects", flush=True)

    features = Features(cond_image=ImageFeature(), image=ImageFeature(), caption=Value("string"),
                        cls=Value("string"), path=Value("string"))
    Dataset.from_dict(dict(rows), features=features).to_parquet(str(out / "data" / "train-00000-of-00001.parquet"))
    with open(out / "holdout.jsonl", "w") as f:
        for h in holdout:
            f.write(json.dumps(h) + "\n")
    (out / "prompt.txt").write_text(PROMPT + "\n")
    print(f"wrote {len(rows['caption'])} train pairs and {len(holdout)} holdout views to {out}")


if __name__ == "__main__":
    main()
