"""Build the full (inspection image -> mask) training manifest for train_qwen_mask_lora_stream.py.

Unlike build_realiad_mask_pairs.py (a small parquet subset that the diffusers script preloads into RAM), this writes
file paths only, so the whole of RealIAD fits:
    * every anomalous view of the test split (target = its binarized GT mask), and
    * every normal view of the train split (target = an all-black mask, `mask: null`),
except views of held-out objects. The held-out anomalous objects are the ones of an existing pilot holdout
(`--pilot`, so results stay comparable with the pilot LoRA), plus `--extra_holdout` objects per class.
Normal test views are never used for training, so `--normal_holdout_per_class` of them (from distinct OK objects)
are added to the holdout to measure whether the model returns an empty mask.

Outputs in --out:
    train.jsonl      {"cond": <image path>, "mask": <mask path or null>, "cls": ...}
    holdout.jsonl    same format as build_realiad_mask_pairs.py (paths relative to --out, `mask: null` for normal)
    holdout_extra.jsonl  all anomalous views of the extra held-out objects (absolute paths)
    prompt.txt

Example:
    python src/experimental/build_realiad_mask_manifest.py --out outputs/realiad_mask_pairs/full
"""
import argparse
import importlib.util
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "realiad", Path(__file__).resolve().parents[1] / "datasets" / "realiad.py")
_realiad = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_realiad)
RealIAD, REALIAD_CLASSES = _realiad.RealIAD, _realiad.REALIAD_CLASSES

PROMPT = (
    "Produce the anomaly segmentation mask for this inspection image. Mark defective pixels white and all other "
    "pixels black. Preserve their exact positions. Return an opaque mask."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/realiad")
    parser.add_argument("--meta_dir", default="data/realiad_meta/realiad_jsons")
    parser.add_argument("--classes", nargs="+", default=REALIAD_CLASSES)
    parser.add_argument("--pilot", default="outputs/realiad_mask_pairs/pilot50")
    parser.add_argument("--extra_holdout", type=int, default=8, help="extra held-out anomalous objects per class")
    parser.add_argument("--normal_holdout_per_class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/realiad_mask_pairs/full")
    args = parser.parse_args()

    out, pilot = Path(args.out), Path(args.pilot)
    (out / "holdout").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    pilot_holdout = [json.loads(line) for line in open(pilot / "holdout.jsonl")]
    pilot_objects = {Path(h["path"]).parent for h in pilot_holdout}

    train, holdout, extra = [], [], []
    for h in pilot_holdout:
        for key in ("image", "mask"):
            shutil.copy(pilot / h[key], out / h[key])
        holdout.append(h)

    for cls in args.classes:
        test = RealIAD(args.data_root, cls, 1024, "test", args.meta_dir)
        by_object = defaultdict(list)
        for i in test.anom_indices:
            by_object[Path(test.img_files[i]).parent].append(i)
        objects = sorted(o for o in by_object if o not in pilot_objects)
        rng.shuffle(objects)
        held = set(objects[:args.extra_holdout])
        for obj in sorted(by_object):
            for i in by_object[obj]:
                row = dict(cond=test.img_files[i], mask=test.masks[i], cls=cls)
                if obj in pilot_objects:
                    continue
                (extra if obj in held else train).append(row)

        # normal holdout: views of fully normal test objects (OK folders), one view per object
        ok_objects = defaultdict(list)
        for i in test.normal_indices:
            if test.labels_str[i] == "OK" and Path(test.img_files[i]).parts[-3] == "OK":
                ok_objects[Path(test.img_files[i]).parent].append(i)
        for obj in rng.sample(sorted(ok_objects), args.normal_holdout_per_class):
            i = rng.choice(ok_objects[obj])
            stem = f"{cls}__{Path(test.img_files[i]).stem}"
            shutil.copy(test.img_files[i], out / "holdout" / f"{stem}{Path(test.img_files[i]).suffix}")
            holdout.append(dict(cls=cls, path=test.img_files[i],
                                image=f"holdout/{stem}{Path(test.img_files[i]).suffix}", mask=None))

        n_normal = 0
        for path in RealIAD(args.data_root, cls, 1024, "train", args.meta_dir).img_files:
            train.append(dict(cond=path, mask=None, cls=cls))
            n_normal += 1
        print(f"{cls}: {len(objects) - len(held)} anomalous train objects, {len(held)} extra held out, "
              f"{n_normal} normal train views", flush=True)

    for name, rows in (("train", train), ("holdout", holdout), ("holdout_extra", extra)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (out / "prompt.txt").write_text(PROMPT + "\n")
    n_anom = sum(r["mask"] is not None for r in train)
    print(f"train: {len(train)} views ({n_anom} anomalous, {len(train) - n_anom} normal); "
          f"holdout: {len(holdout)} views; holdout_extra: {len(extra)} views -> {out}")


if __name__ == "__main__":
    main()
