#!/usr/bin/env python3
"""
Phase 8A: Build D5_HN dataset for YOLO hard-negative fine-tuning.

Purpose
-------
Create a new 2-class YOLO detection dataset by copying the frozen D5-clean
train/valid/test splits and adding WBF/CLIP-mined non-weapon crops to the
TRAIN split only as empty-label background images.

Important
---------
- Detector classes remain: 0=knife, 1=pistol.
- Non-weapon is NOT added as a third detector class.
- Hard-negative crops get empty .txt labels.
- Validation and test splits are copied unchanged from the original YAML.

Example
-------
python phase8_build_d5_hn_dataset.py \
  --src_yaml /home/inspur/ishrai/yolo_server.yaml \
  --hn_dir /home/inspur/ishrai/clip_dataset_wbfonly/train/non_weapon \
  --out_dir /home/inspur/ishrai/D5_HN \
  --copy_mode copy \
  --overwrite
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser("Build D5_HN hard-negative YOLO dataset")
    p.add_argument("--src_yaml", required=True, help="Frozen D5-clean YOLO YAML, e.g. /home/inspur/ishrai/yolo_server.yaml")
    p.add_argument("--hn_dir", required=True, help="Training hard-negative crop directory, e.g. clip_dataset_wbfonly/train/non_weapon")
    p.add_argument("--out_dir", required=True, help="Output D5_HN dataset directory")
    p.add_argument("--copy_mode", choices=["copy", "symlink"], default="copy", help="copy is safest; symlink saves disk space")
    p.add_argument("--max_hn", type=int, default=None, help="Optional maximum number of hard negatives to add")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="Remove out_dir first if it exists")
    p.add_argument("--dry_run", action="store_true", help="Print planned actions without copying files")
    return p.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return data


def yaml_root(src_yaml: Path, data: dict) -> Path:
    root = data.get("path", None)
    if root is None:
        return src_yaml.parent.resolve()
    root_path = Path(str(root))
    if not root_path.is_absolute():
        root_path = (src_yaml.parent / root_path).resolve()
    return root_path


def resolve_split(data: dict, root: Path, key_candidates: List[str]) -> Optional[Path]:
    value = None
    for key in key_candidates:
        if key in data:
            value = data[key]
            break
    if value is None:
        return None
    if isinstance(value, list):
        raise ValueError(f"YAML split {key_candidates} is a list. This script expects a directory or text-file path.")
    split_path = Path(str(value))
    if not split_path.is_absolute():
        split_path = (root / split_path).resolve()
    return split_path


def iter_images_from_split(split_path: Path) -> List[Path]:
    if split_path.is_dir():
        imgs = [p for p in split_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
        return sorted(imgs)
    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        base = split_path.parent
        imgs = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = (base / p).resolve()
            if p.suffix.lower() in IMG_EXTS:
                imgs.append(p)
        return sorted(imgs)
    raise ValueError(f"Unsupported split path. Expected image directory or txt list: {split_path}")


def infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    # Prefer the usual YOLO layout: .../images/split/file.jpg -> .../labels/split/file.txt
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    # Fallback: sibling labels directory next to image split directory
    return image_path.with_suffix(".txt")


def safe_copy_or_link(src: Path, dst: Path, mode: str, dry_run: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def unique_dst_name(prefix: str, idx: int, src: Path) -> str:
    stem = src.stem
    # Keep names manageable and deterministic.
    clean_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)[:80]
    return f"{prefix}_{idx:06d}_{clean_stem}{src.suffix.lower()}"


def count_label_instances(label_file: Path) -> Tuple[int, int, int]:
    knife = 0
    pistol = 0
    total = 0
    if not label_file.exists():
        return 0, 0, 0
    txt = label_file.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return 0, 0, 0
    for line in txt.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        total += 1
        if cls == 0:
            knife += 1
        elif cls == 1:
            pistol += 1
    return knife, pistol, total


def copy_split(split_name: str, image_paths: List[Path], out_dir: Path, copy_mode: str, dry_run: bool) -> Dict[str, int]:
    out_img_dir = out_dir / "images" / split_name
    out_lbl_dir = out_dir / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "images": 0,
        "labels_present": 0,
        "labels_missing_created_empty": 0,
        "knife_instances": 0,
        "pistol_instances": 0,
        "total_instances": 0,
    }

    for i, img in enumerate(image_paths, start=1):
        dst_img = out_img_dir / unique_dst_name(split_name, i, img)
        dst_lbl = out_lbl_dir / (dst_img.stem + ".txt")
        src_lbl = infer_label_path(img)

        safe_copy_or_link(img, dst_img, copy_mode, dry_run=dry_run)
        if src_lbl.exists():
            safe_copy_or_link(src_lbl, dst_lbl, copy_mode, dry_run=dry_run)
            stats["labels_present"] += 1
            k, p, t = count_label_instances(src_lbl)
            stats["knife_instances"] += k
            stats["pistol_instances"] += p
            stats["total_instances"] += t
        else:
            if not dry_run:
                dst_lbl.write_text("", encoding="utf-8")
            stats["labels_missing_created_empty"] += 1

        stats["images"] += 1

    return stats


def add_hard_negatives(hn_paths: List[Path], out_dir: Path, copy_mode: str, dry_run: bool) -> Dict[str, int]:
    out_img_dir = out_dir / "images" / "train"
    out_lbl_dir = out_dir / "labels" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(hn_paths, start=1):
        dst_img = out_img_dir / unique_dst_name("hn_nonweapon", i, img)
        dst_lbl = out_lbl_dir / (dst_img.stem + ".txt")
        safe_copy_or_link(img, dst_img, copy_mode, dry_run=dry_run)
        if not dry_run:
            dst_lbl.write_text("", encoding="utf-8")

    return {
        "hard_negative_images_added_to_train": len(hn_paths),
        "hard_negative_empty_labels_created": len(hn_paths),
    }


def write_yaml(out_dir: Path, out_yaml: Path, names: Dict[int, str], dry_run: bool):
    clean_names = {0: "knife", 1: "pistol"}
    # Preserve names only if they clearly contain knife/pistol in positions 0 and 1.
    if isinstance(names, dict):
        n0 = str(names.get(0, names.get("0", "knife"))).lower()
        n1 = str(names.get(1, names.get("1", "pistol"))).lower()
        if "knife" in n0 and ("pistol" in n1 or "gun" in n1 or "handgun" in n1):
            clean_names = {0: str(names.get(0, names.get("0"))), 1: str(names.get(1, names.get("1")))}
    elif isinstance(names, list) and len(names) >= 2:
        n0 = str(names[0]).lower()
        n1 = str(names[1]).lower()
        if "knife" in n0 and ("pistol" in n1 or "gun" in n1 or "handgun" in n1):
            clean_names = {0: str(names[0]), 1: str(names[1])}

    y = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": 2,
        "names": clean_names,
    }
    if not dry_run:
        with out_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(y, f, sort_keys=False, allow_unicode=True)


def write_summary(out_dir: Path, src_yaml: Path, hn_dir: Path, copy_mode: str, split_stats: Dict[str, dict], hn_stats: dict, dry_run: bool):
    summary = {
        "source_yaml": str(src_yaml),
        "hard_negative_source_dir": str(hn_dir),
        "output_dir": str(out_dir),
        "copy_mode": copy_mode,
        "detector_classes": {"0": "knife", "1": "pistol"},
        "non_weapon_handling": "added to train only as empty-label background images; not a detector class",
        "splits": split_stats,
        "hard_negatives": hn_stats,
        "dry_run": dry_run,
    }
    if not dry_run:
        with (out_dir / "phase8_d5_hn_dataset_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with (out_dir / "phase8_d5_hn_dataset_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["section", "key", "value"])
            for split, stats in split_stats.items():
                for k, v in stats.items():
                    writer.writerow([split, k, v])
            for k, v in hn_stats.items():
                writer.writerow(["hard_negatives", k, v])
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    src_yaml = Path(args.src_yaml).resolve()
    hn_dir = Path(args.hn_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not src_yaml.exists():
        raise FileNotFoundError(f"Source YAML not found: {src_yaml}")
    if not hn_dir.exists() or not hn_dir.is_dir():
        raise FileNotFoundError(f"Hard-negative directory not found: {hn_dir}")

    if out_dir.exists() and args.overwrite and not args.dry_run:
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output directory exists and is not empty. Use --overwrite or a new --out_dir: {out_dir}")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    data = load_yaml(src_yaml)
    root = yaml_root(src_yaml, data)

    split_paths = {
        "train": resolve_split(data, root, ["train"]),
        "valid": resolve_split(data, root, ["val", "valid"]),
        "test": resolve_split(data, root, ["test"]),
    }
    missing = [k for k, v in split_paths.items() if v is None]
    if missing:
        raise ValueError(f"Missing split(s) in YAML: {missing}. Need train, val/valid, and test.")

    print("Resolved split paths:")
    for k, v in split_paths.items():
        print(f"  {k}: {v}")

    split_images = {k: iter_images_from_split(v) for k, v in split_paths.items()}
    for k, imgs in split_images.items():
        if not imgs:
            raise ValueError(f"No images found for split {k}: {split_paths[k]}")
        print(f"Found {len(imgs)} images for {k}")

    hn_paths = sorted([p for p in hn_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if args.max_hn is not None:
        import random
        rng = random.Random(args.seed)
        hn_paths = hn_paths[:]
        rng.shuffle(hn_paths)
        hn_paths = sorted(hn_paths[: args.max_hn])
    if not hn_paths:
        raise ValueError(f"No hard-negative images found in: {hn_dir}")
    print(f"Found {len(hn_paths)} hard-negative crop images")

    split_stats = {}
    for split_name in ["train", "valid", "test"]:
        print(f"\nCopying split: {split_name}")
        split_stats[split_name] = copy_split(
            split_name=split_name,
            image_paths=split_images[split_name],
            out_dir=out_dir,
            copy_mode=args.copy_mode,
            dry_run=args.dry_run,
        )

    print("\nAdding hard negatives to train split only")
    hn_stats = add_hard_negatives(hn_paths, out_dir, args.copy_mode, dry_run=args.dry_run)

    out_yaml = out_dir / "d5_hn.yaml"
    write_yaml(out_dir, out_yaml, data.get("names", {0: "knife", 1: "pistol"}), dry_run=args.dry_run)

    write_summary(out_dir, src_yaml, hn_dir, args.copy_mode, split_stats, hn_stats, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nDONE")
        print(f"New dataset YAML: {out_yaml}")
        print("Detector classes remain: 0=knife, 1=pistol")
        print("Hard negatives were added only to images/train with empty labels/train files.")


if __name__ == "__main__":
    main()
