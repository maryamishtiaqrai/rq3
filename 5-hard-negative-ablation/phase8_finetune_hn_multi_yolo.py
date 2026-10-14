#!/usr/bin/env python3
"""
Phase 8B: Fine-tune multiple YOLO detectors on D5_HN.

This is based on the user's multi-model training pattern, but names the runs as
hard-negative fine-tuning and exposes common fine-tuning parameters.

Expected weights in --weights_dir:
  y8x_best.pt
  y8l_best.pt
  y26n_best.pt
  y26s_best.pt

Example:
python phase8_finetune_hn_multi_yolo.py \
  --models y8x y8l y26n y26s \
  --weights_dir /home/inspur/ishrai/all_togather \
  --data /home/inspur/ishrai/D5_HN/d5_hn.yaml \
  --epochs 60 --patience 20 --batch 16 --device 0 \
  --project /home/inspur/ishrai/runs/detect \
  --name_prefix weapon_d5hn_ft
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def sanitize_name(s: str) -> str:
    base = os.path.basename(s).replace(".pt", "")
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    return base[:80]


def parse_args():
    p = argparse.ArgumentParser("Phase 8 hard-negative fine-tuning for multiple YOLO models")
    p.add_argument("--models", type=str, nargs="+", required=False, help="Model IDs, e.g. y8x y8l y26n y26s")
    p.add_argument("--weights_dir", type=str, required=False, help="Directory containing *_best.pt weights")
    p.add_argument("--data", type=str, required=False, help="D5_HN YAML file path")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--project", type=str, default="/home/inspur/ishrai/runs/detect")
    p.add_argument("--name_prefix", type=str, default="weapon_d5hn_ft")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optimizer", type=str, default="auto", help="Ultralytics optimizer, e.g. auto, SGD, AdamW")
    p.add_argument("--lr0", type=float, default=1e-4, help="Lower learning rate for fine-tuning")
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--cos_lr", action="store_true")
    p.add_argument("--close_mosaic", type=int, default=10)
    p.add_argument("--save_period", type=int, default=-1)
    p.add_argument("--exist_ok", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Print planned runs only")

    p.add_argument("--resume", action="store_true", help="Resume training from last.pt checkpoints")
    p.add_argument("--resume_dir", type=str, default=None, help="Specific run directory to resume from")
    return p.parse_args()


def find_last_checkpoint(run_dir: Path):
    last_pt = run_dir / "weights" / "last.pt"
    return last_pt if last_pt.exists() else None


def find_run_dirs(project_dir: Path, name_prefix: str):
    if not project_dir.exists():
        return []
    return sorted([p for p in project_dir.iterdir() if p.is_dir() and p.name.startswith(name_prefix)])


def write_manifest(project_dir: Path, run_rows, name_prefix: str):
    project_dir.mkdir(parents=True, exist_ok=True)
    csv_path = project_dir / f"{name_prefix}_manifest.csv"
    json_path = project_dir / f"{name_prefix}_manifest.json"
    if run_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(run_rows, f, indent=2)
    print(f"Saved manifest CSV : {csv_path}")
    print(f"Saved manifest JSON: {json_path}")


def main():
    args = parse_args()
    project_dir = Path(args.project).resolve()

    if args.resume:
        print("\n" + "=" * 80)
        print("RESUME MODE")
        print("=" * 80)
        if args.resume_dir:
            run_dir = Path(args.resume_dir).resolve()
            last_pt = find_last_checkpoint(run_dir)
            if not last_pt:
                raise FileNotFoundError(f"No last.pt found in: {run_dir}")
            print(f"Resuming: {last_pt}")
            model = YOLO(str(last_pt))
            model.train(resume=True)
            return

        run_dirs = find_run_dirs(project_dir, args.name_prefix)
        if not run_dirs:
            raise FileNotFoundError(f"No run folders found with prefix {args.name_prefix} in {project_dir}")
        print(f"Found {len(run_dirs)} run folders")
        for run_dir in run_dirs:
            last_pt = find_last_checkpoint(run_dir)
            if not last_pt:
                print(f"Skipping, no last.pt: {run_dir}")
                continue
            print("\n" + "=" * 80)
            print(f"Resuming: {run_dir}")
            print(f"Checkpoint: {last_pt}")
            print("=" * 80)
            model = YOLO(str(last_pt))
            model.train(resume=True)
        return

    if not args.models:
        raise ValueError("--models is required when not in resume mode")
    if not args.weights_dir:
        raise ValueError("--weights_dir is required when not in resume mode")
    if not args.data:
        raise ValueError("--data is required when not in resume mode")

    data_path = Path(args.data).resolve()
    weights_dir = Path(args.weights_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_rows = []

    print("\n" + "=" * 80)
    print("PHASE 8 HARD-NEGATIVE FINE-TUNING")
    print("=" * 80)
    print(f"Data YAML     : {data_path}")
    print(f"Weights dir   : {weights_dir}")
    print(f"Project       : {project_dir}")
    print(f"Models        : {args.models}")
    print(f"Epochs        : {args.epochs}")
    print(f"Patience      : {args.patience}")
    print(f"LR0           : {args.lr0}")
    print("=" * 80)

    for model_id in args.models:
        weight_file = weights_dir / f"{model_id}_best.pt"
        if not weight_file.exists():
            raise FileNotFoundError(f"Weights not found: {weight_file}")

        run_name = f"{args.name_prefix}_{sanitize_name(model_id)}_from_D5best_to_D5HN_{ts}"
        row = {
            "model_id": model_id,
            "start_weights": str(weight_file),
            "data_yaml": str(data_path),
            "project": str(project_dir),
            "run_name": run_name,
            "epochs": args.epochs,
            "patience": args.patience,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "seed": args.seed,
            "status": "planned" if args.dry_run else "started",
        }
        run_rows.append(row)

        print("\n" + "=" * 80)
        print(f"Model ID      : {model_id}")
        print(f"Start weights : {weight_file}")
        print(f"Run name      : {run_name}")
        print("=" * 80)

        if args.dry_run:
            continue

        model = YOLO(str(weight_file))
        model.info()
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            patience=args.patience,
            project=str(project_dir),
            name=run_name,
            workers=args.workers,
            seed=args.seed,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            cos_lr=args.cos_lr,
            close_mosaic=args.close_mosaic,
            save_period=args.save_period,
            exist_ok=args.exist_ok,
        )
        row["status"] = "finished"
        row["run_dir"] = str(project_dir / run_name)
        row["best_pt_expected"] = str(project_dir / run_name / "weights" / "best.pt")
        row["last_pt_expected"] = str(project_dir / run_name / "weights" / "last.pt")
        write_manifest(project_dir, run_rows, args.name_prefix)

    write_manifest(project_dir, run_rows, args.name_prefix)
    print("\nDONE: Phase 8 fine-tuning runs completed or planned.")


if __name__ == "__main__":
    main()
