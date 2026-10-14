#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6: Single-YOLO verifier control experiment for all detectors.

Purpose
-------
This phase checks whether the CLIP / YOLO-cls verifier helps only after WBF,
or whether the same semantic-verification idea also helps single YOLO detectors.

It evaluates four frozen single detectors:
  - y8x_best.pt
  - y8l_best.pt
  - y26n_best.pt
  - y26s_best.pt

For each detector, it tests:
  1) YOLO threshold curve
  2) verifier filter-only methods
  3) verifier recovery-only methods
  4) verifier filter+recovery entropy method
  5) optional semantic score-rescoring methods

Important protocol
------------------
- Validation split is used for selecting thresholds/configurations.
- Test split is used only after validation selection.
- Candidate detections and verifier scores are saved, so expensive inference does not need to be repeated.
- Numeric detector class IDs and verifier class IDs are never compared directly; class names are used.

Default server paths
--------------------
Detection models:
  /home/inspur/ishrai/all_togather/y8x_best.pt
  /home/inspur/ishrai/all_togather/y8l_best.pt
  /home/inspur/ishrai/all_togather/y26n_best.pt
  /home/inspur/ishrai/all_togather/y26s_best.pt

CLIP verifier:
  /home/inspur/ishrai/all_togather/clip_verifier/best_clip_finetuned.pth

YOLO-cls verifier:
  /home/inspur/ishrai/all_togather/yolo_verifier/YOLOv8l_cls_best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import yaml

try:
    import open_clip
except Exception as e:
    open_clip = None
    OPEN_CLIP_IMPORT_ERROR = e
else:
    OPEN_CLIP_IMPORT_ERROR = None

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = e
else:
    ULTRALYTICS_IMPORT_ERROR = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
WEAPON_CLASSES = ["knife", "pistol"]
UNIFIED_CLASSES = ["knife", "pistol", "non_weapon"]
DEFAULT_DET_NAMES = {0: "knife", 1: "pistol"}
EPS = 1e-12


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_json_dump(obj, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_float_grid(text: str) -> List[float]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def canonical_name(x) -> str:
    s = str(x).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    if s in ["background", "nonweapon", "non_weapon_object", "non_weapon_objects", "negative", "bg"]:
        return "non_weapon"
    return s


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def cuda_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, EPS, 1.0)
    ent = -np.sum(probs * np.log(probs), axis=1) / math.log(probs.shape[1])
    return ent.astype(np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(float(a[0]), float(b[0]))
    yy1 = max(float(a[1]), float(b[1]))
    xx2 = min(float(a[2]), float(b[2]))
    yy2 = min(float(a[3]), float(b[3]))
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def average_precision_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def safe_write_excel(xlsx_path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
            ws = writer.book[safe_name]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col[:2000]:
                    try:
                        max_len = max(max_len, len(str(cell.value)) if cell.value is not None else 0)
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 40)


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

def resolve_path(path_value: str, root: Path, yaml_parent: Path) -> Path:
    path_value = os.path.expanduser(str(path_value))
    p = Path(path_value)
    if p.is_absolute():
        return p
    if root and str(root) not in {".", ""}:
        return (root / p).resolve()
    return (yaml_parent / p).resolve()


def load_split_from_yaml(data_yaml: str, split: str) -> Tuple[Path, Path, Dict[int, str]]:
    yaml_path = Path(data_yaml).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Data yaml not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root_val = data.get("path", "")
    root = resolve_path(root_val, Path(""), yaml_path.parent) if root_val else yaml_path.parent

    if split in {"valid", "val"}:
        key = "val" if "val" in data else "valid"
    else:
        key = split
    if key not in data:
        raise KeyError(f"Split key '{key}' not found in {yaml_path}. Available keys: {list(data.keys())}")

    img_value = data[key]
    if isinstance(img_value, list):
        if len(img_value) != 1:
            raise ValueError("This script expects one image directory per split. Found multiple entries.")
        img_value = img_value[0]

    img_dir = resolve_path(str(img_value), root, yaml_path.parent)
    if not img_dir.exists():
        raise FileNotFoundError(f"Image split path does not exist: {img_dir}")

    parts = list(img_dir.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        label_parts = parts.copy()
        label_parts[idx] = "labels"
        label_dir = Path(*label_parts)
    elif img_dir.parent.name == "images":
        label_dir = img_dir.parent.parent / "labels" / img_dir.name
    else:
        label_dir = img_dir.parent / "labels"

    if not label_dir.exists():
        raise FileNotFoundError(f"Inferred label directory not found: {label_dir}\nImage directory: {img_dir}")

    names_raw = data.get("names", DEFAULT_DET_NAMES)
    if isinstance(names_raw, dict):
        names = {int(k): canonical_name(v) for k, v in names_raw.items()}
    else:
        names = {i: canonical_name(v) for i, v in enumerate(names_raw)}

    return img_dir, label_dir, names


def list_images(img_dir: Path, limit: int = 0) -> List[Path]:
    images = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if limit and limit > 0:
        images = images[:limit]
    if not images:
        raise RuntimeError(f"No images found in {img_dir}")
    return images


def image_id_from_path(img_path: Path, img_dir: Path) -> str:
    return str(img_path.relative_to(img_dir)).replace("\\", "/")


def label_path_for_image(img_path: Path, img_dir: Path, label_dir: Path) -> Path:
    rel = img_path.relative_to(img_dir).with_suffix(".txt")
    return label_dir / rel


def read_yolo_label(label_path: Path, class_names: Dict[int, str]) -> List[Dict]:
    rows: List[Dict] = []
    if not label_path.exists():
        return rows
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            x1 = max(0.0, cx - w / 2.0)
            y1 = max(0.0, cy - h / 2.0)
            x2 = min(1.0, cx + w / 2.0)
            y2 = min(1.0, cy + h / 2.0)
            cname = canonical_name(class_names.get(cls, str(cls)))
            if cname not in WEAPON_CLASSES:
                continue
            rows.append({
                "class_id": cls,
                "class_name": cname,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    return rows


# -----------------------------------------------------------------------------
# Single-YOLO candidate generation
# -----------------------------------------------------------------------------

def resolve_detector_paths(model_dir: Path, model_keys: List[str]) -> Dict[str, Path]:
    all_paths = {
        "y8x": model_dir / "y8x_best.pt",
        "y8l": model_dir / "y8l_best.pt",
        "y26n": model_dir / "y26n_best.pt",
        "y26s": model_dir / "y26s_best.pt",
    }
    selected = {k: all_paths[k] for k in model_keys}
    missing = [str(p) for p in selected.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing detector model file(s):\n" + "\n".join(missing))
    return selected


def predict_single_yolo(model: YOLO, img_path: Path, args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    cuda_sync(args.device)
    t0 = time.perf_counter()
    results = model.predict(
        source=str(img_path),
        imgsz=args.imgsz,
        conf=args.low_conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
    )
    cuda_sync(args.device)
    elapsed = time.perf_counter() - t0

    r = results[0]
    h, w = r.orig_shape[:2]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=int), elapsed

    xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(float)
    scores = r.boxes.conf.detach().cpu().numpy().astype(float)
    labels = r.boxes.cls.detach().cpu().numpy().astype(int)
    xyxy[:, [0, 2]] /= float(w)
    xyxy[:, [1, 3]] /= float(h)
    xyxy = np.clip(xyxy, 0.0, 1.0)
    return xyxy, scores, labels, elapsed


def generate_candidates_for_split(split: str, args, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if YOLO is None:
        raise RuntimeError(f"ultralytics import failed: {ULTRALYTICS_IMPORT_ERROR}")

    split_dir = out_dir / split
    ensure_dir(split_dir)
    cand_csv = split_dir / f"phase6_{split}_single_yolo_candidates.csv"
    gt_csv = split_dir / f"phase6_{split}_single_yolo_ground_truth.csv"
    timing_csv = split_dir / f"phase6_{split}_single_yolo_timing_by_image.csv"

    if args.reuse_candidates and cand_csv.exists() and gt_csv.exists() and timing_csv.exists():
        print(f"[{split}] Reusing existing candidate files from {split_dir}")
        return pd.read_csv(cand_csv), pd.read_csv(gt_csv), pd.read_csv(timing_csv)

    model_keys = parse_list(args.detectors)
    model_dir = Path(args.model_dir).expanduser().resolve()
    model_paths = resolve_detector_paths(model_dir, model_keys)

    img_dir, label_dir, class_names = load_split_from_yaml(args.data_yaml, split)
    images = list_images(img_dir, limit=args.limit)

    print("=" * 80)
    print(f"Phase 6 candidate generation: {split}")
    print(f"Images      : {len(images)}")
    print(f"Image dir   : {img_dir}")
    print(f"Label dir   : {label_dir}")
    print(f"Class names : {class_names}")
    print("Detectors:")
    for k, p in model_paths.items():
        print(f"  {k}: {p}")
    print("=" * 80)

    # Ground truth is generated once per split.
    gt_rows = []
    for img_path in images:
        image_id = image_id_from_path(img_path, img_dir)
        lab_path = label_path_for_image(img_path, img_dir, label_dir)
        gts = read_yolo_label(lab_path, class_names)
        for g in gts:
            gt_rows.append({
                "split": split,
                "image_id": image_id,
                "image_path": str(img_path),
                "label_path": str(lab_path),
                **g,
            })
    gt_df = pd.DataFrame(gt_rows)
    gt_df.to_csv(gt_csv, index=False)

    cand_rows = []
    timing_rows = []

    for model_key, model_path in model_paths.items():
        print(f"\n[{split}] Loading detector: {model_key} -> {model_path}")
        model = YOLO(str(model_path))
        for img_path in tqdm(images, desc=f"{split}: {model_key}"):
            image_id = image_id_from_path(img_path, img_dir)
            boxes, scores, labels, elapsed = predict_single_yolo(model, img_path, args)
            saved_count = 0
            for box, score, cls in zip(boxes, scores, labels):
                score = float(score)
                if score < args.save_conf:
                    continue
                cls = int(cls)
                cname = canonical_name(class_names.get(cls, str(cls)))
                if cname not in WEAPON_CLASSES:
                    continue
                saved_count += 1
                cand_rows.append({
                    "split": split,
                    "pair": model_key,                      # kept for compatibility with earlier scripts
                    "detector": model_key,
                    "image_id": image_id,
                    "image_path": str(img_path),
                    "class_id": cls,
                    "class_name": cname,
                    "score": score,
                    "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
                    "source_type": "single_yolo_detection",
                    "support_count": 1,
                    "num_boxes_in_cluster": 1,
                    "model_support_indices": model_key,
                })
            timing_rows.append({
                "split": split,
                "pair": model_key,
                "detector": model_key,
                "image_id": image_id,
                "image_path": str(img_path),
                "detector_inference_sec": float(elapsed),
                "total_sec_single_yolo": float(elapsed),
                # compatibility name used by earlier timing functions
                "total_sec_model1_model2_wbf": float(elapsed),
                "num_candidates_raw": int(len(scores)),
                "num_candidates_saved": int(saved_count),
            })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cand_df = pd.DataFrame(cand_rows)
    timing_df = pd.DataFrame(timing_rows)
    cand_df.to_csv(cand_csv, index=False)
    timing_df.to_csv(timing_csv, index=False)

    # Excel files are small enough here and useful for audit.
    cand_xlsx = split_dir / f"phase6_{split}_single_yolo_candidates.xlsx"
    gt_xlsx = split_dir / f"phase6_{split}_single_yolo_ground_truth.xlsx"
    timing_xlsx = split_dir / f"phase6_{split}_single_yolo_timing_by_image.xlsx"
    if len(cand_df) <= 1_000_000:
        cand_df.to_excel(cand_xlsx, index=False)
    gt_df.to_excel(gt_xlsx, index=False)
    timing_df.to_excel(timing_xlsx, index=False)

    print(f"[{split}] Saved candidates: {cand_csv}")
    print(f"[{split}] Saved ground truth: {gt_csv}")
    print(f"[{split}] Saved timing: {timing_csv}")
    return cand_df, gt_df, timing_df


# -----------------------------------------------------------------------------
# Crop extraction and verifier scoring
# -----------------------------------------------------------------------------

def maybe_abs_to_norm_box(row, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, [row["x1"], row["y1"], row["x2"], row["y2"]])
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 2.0:
        x1, x2 = x1 / img_w, x2 / img_w
        y1, y2 = y1 / img_h, y2 / img_h
    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    y2 = min(max(y2, 0.0), 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def crop_candidate(image_bgr: np.ndarray, box_norm: Tuple[float, float, float, float], pad: float = 0.05,
                   min_size: int = 4) -> Optional[Image.Image]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box_norm
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    x1p = max(0.0, x1 - pad * bw)
    y1p = max(0.0, y1 - pad * bh)
    x2p = min(1.0, x2 + pad * bw)
    y2p = min(1.0, y2 + pad * bh)
    ix1 = int(round(x1p * w))
    iy1 = int(round(y1p * h))
    ix2 = int(round(x2p * w))
    iy2 = int(round(y2p * h))
    ix1, iy1 = max(0, ix1), max(0, iy1)
    ix2, iy2 = min(w, ix2), min(h, iy2)
    if ix2 - ix1 < min_size or iy2 - iy1 < min_size:
        return None
    crop_bgr = image_bgr[iy1:iy2, ix1:ix2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def add_detector_class_probability(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    for c in [f"{prefix}_prob_knife", f"{prefix}_prob_pistol", f"{prefix}_prob_non_weapon", f"{prefix}_pred_name"]:
        if c not in df.columns:
            df[c] = np.nan
    probs = []
    for cname, pk, pp, pn in zip(df["class_name"], df[f"{prefix}_prob_knife"], df[f"{prefix}_prob_pistol"], df[f"{prefix}_prob_non_weapon"]):
        c = canonical_name(cname)
        if c == "knife":
            probs.append(pk)
        elif c == "pistol":
            probs.append(pp)
        else:
            probs.append(np.nan)
    df[f"{prefix}_prob_detector_class"] = probs
    df[f"{prefix}_same_as_detector"] = (
        df[f"{prefix}_pred_name"].map(canonical_name) == df["class_name"].map(canonical_name)
    )
    return df


def load_clip_verifier(args, device: str):
    if open_clip is None:
        raise RuntimeError(f"open_clip import failed: {OPEN_CLIP_IMPORT_ERROR}")
    pretrained_arg = None
    if args.clip_pretrained and str(args.clip_pretrained).lower() not in ["none", "null", ""]:
        if Path(args.clip_pretrained).exists():
            pretrained_arg = args.clip_pretrained
        else:
            print(f"WARNING: CLIP pretrained file not found: {args.clip_pretrained}")
            print("Proceeding with pretrained=None and loading the fine-tuned checkpoint afterward.")

    print(f"Loading CLIP model: {args.clip_model_name}")
    print(f"CLIP pretrained argument: {pretrained_arg}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model_name,
        pretrained=pretrained_arg,
        device=device,
    )

    ckpt_path = Path(args.clip_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"CLIP checkpoint not found: {ckpt_path}")
    print(f"Loading fine-tuned CLIP checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        else:
            state = ckpt
    else:
        state = ckpt
    clean_state = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        clean_state[nk] = v
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(f"CLIP load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 20 or len(unexpected) > 20:
        print("WARNING: many missing/unexpected CLIP keys. Check model/checkpoint compatibility.")
        print("First missing:", list(missing)[:10])
        print("First unexpected:", list(unexpected)[:10])

    model.eval()
    tokenizer = open_clip.get_tokenizer(args.clip_model_name)
    prompts = [
        "a cropped image of a knife",
        "a cropped image of a pistol",
        "a cropped image of a non-weapon object",
    ]
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return model, preprocess, text_features


def score_clip_batch(model, preprocess, text_features, crops: List[Image.Image], device: str) -> np.ndarray:
    ims = torch.stack([preprocess(im) for im in crops], dim=0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(ims)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else torch.tensor(100.0, device=device)
        logits = logit_scale * image_features @ text_features.T
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    return probs.astype(np.float32)


def load_yolo_cls(args):
    if YOLO is None:
        raise RuntimeError(f"ultralytics import failed: {ULTRALYTICS_IMPORT_ERROR}")
    ckpt_path = Path(args.yolo_cls_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"YOLO classification checkpoint not found: {ckpt_path}")
    print(f"Loading YOLO classification verifier: {ckpt_path}")
    model = YOLO(str(ckpt_path))
    names = model.names
    print("YOLO-cls class names:", names)
    canon_to_idx = {}
    for k, v in names.items():
        canon = canonical_name(v)
        if canon in UNIFIED_CLASSES:
            canon_to_idx[canon] = int(k)
    missing = [c for c in UNIFIED_CLASSES if c not in canon_to_idx]
    if missing:
        raise RuntimeError(f"YOLO-cls names cannot be mapped to {UNIFIED_CLASSES}. Missing: {missing}; names={names}")
    return model, canon_to_idx


def score_yolo_cls_batch(model, canon_to_idx: Dict[str, int], crops: List[Image.Image], args) -> np.ndarray:
    sources = [np.asarray(im.convert("RGB")) for im in crops]
    results = model.predict(
        source=sources,
        imgsz=args.yolo_cls_imgsz,
        device=args.device,
        verbose=False,
        batch=len(sources),
    )
    out = []
    for r in results:
        if r.probs is None:
            raise RuntimeError("YOLO classification result has no r.probs. Check classifier checkpoint.")
        p = r.probs.data.detach().cpu().numpy().astype(np.float32)
        aligned = np.zeros(3, dtype=np.float32)
        aligned[0] = p[canon_to_idx["knife"]]
        aligned[1] = p[canon_to_idx["pistol"]]
        aligned[2] = p[canon_to_idx["non_weapon"]]
        out.append(aligned)
    return np.vstack(out).astype(np.float32)


def run_verifier_scoring_for_split(split: str, cand_df: pd.DataFrame, args, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_dir = out_dir / split
    ensure_dir(split_dir)
    scored_csv = split_dir / f"phase6_{split}_candidate_verifier_scores.csv"
    scored_xlsx = split_dir / f"phase6_{split}_candidate_verifier_scores.xlsx"
    timing_csv = split_dir / f"phase6_{split}_verifier_batch_timing.csv"
    timing_xlsx = split_dir / f"phase6_{split}_verifier_batch_timing.xlsx"

    if args.reuse_scores and scored_csv.exists() and timing_csv.exists():
        print(f"[{split}] Reusing existing verifier scores from {scored_csv}")
        return pd.read_csv(scored_csv), pd.read_csv(timing_csv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    clip_model, clip_preprocess, clip_text_features = load_clip_verifier(args, device)
    yolo_cls_model, yolo_cls_map = load_yolo_cls(args)

    scored_rows = []
    timing_rows = []
    print(f"[{split}] Scoring {len(cand_df)} single-YOLO candidate crops.")

    batch_crops: List[Image.Image] = []
    batch_meta: List[Dict] = []

    def flush_batch():
        nonlocal batch_crops, batch_meta, scored_rows, timing_rows
        if not batch_crops:
            return
        n = len(batch_crops)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        clip_probs = score_clip_batch(clip_model, clip_preprocess, clip_text_features, batch_crops, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        yolo_probs = score_yolo_cls_batch(yolo_cls_model, yolo_cls_map, batch_crops, args)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        clip_sec = t1 - t0
        yolo_sec = t3 - t2
        clip_per_crop = clip_sec / max(n, 1)
        yolo_per_crop = yolo_sec / max(n, 1)

        clip_ent = normalized_entropy(clip_probs)
        yolo_ent = normalized_entropy(yolo_probs)
        clip_pred = np.argmax(clip_probs, axis=1)
        yolo_pred = np.argmax(yolo_probs, axis=1)

        for j in range(n):
            base = dict(batch_meta[j])
            base.update({
                "crop_valid": True,
                "clip_prob_knife": float(clip_probs[j, 0]),
                "clip_prob_pistol": float(clip_probs[j, 1]),
                "clip_prob_non_weapon": float(clip_probs[j, 2]),
                "clip_pred_id": int(clip_pred[j]),
                "clip_pred_name": UNIFIED_CLASSES[int(clip_pred[j])],
                "clip_entropy": float(clip_ent[j]),
                "clip_inference_sec_est": float(clip_per_crop),
                "yolo_prob_knife": float(yolo_probs[j, 0]),
                "yolo_prob_pistol": float(yolo_probs[j, 1]),
                "yolo_prob_non_weapon": float(yolo_probs[j, 2]),
                "yolo_pred_id": int(yolo_pred[j]),
                "yolo_pred_name": UNIFIED_CLASSES[int(yolo_pred[j])],
                "yolo_entropy": float(yolo_ent[j]),
                "yolo_inference_sec_est": float(yolo_per_crop),
            })
            scored_rows.append(base)

        timing_rows.append({
            "split": split,
            "batch_size": n,
            "clip_batch_sec": clip_sec,
            "clip_sec_per_crop": clip_per_crop,
            "clip_crops_per_sec": n / clip_sec if clip_sec > 0 else np.nan,
            "yolo_batch_sec": yolo_sec,
            "yolo_sec_per_crop": yolo_per_crop,
            "yolo_crops_per_sec": n / yolo_sec if yolo_sec > 0 else np.nan,
        })
        batch_crops, batch_meta = [], []

    for image_path, g in tqdm(cand_df.groupby("image_path", sort=False), total=cand_df["image_path"].nunique(), desc=f"{split}: images"):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"WARNING: cannot read image: {image_path}")
            for _, row in g.iterrows():
                base = row.to_dict()
                base.update({"crop_valid": False})
                scored_rows.append(base)
            continue
        h, w = image_bgr.shape[:2]
        for _, row in g.iterrows():
            box = maybe_abs_to_norm_box(row, w, h)
            crop = crop_candidate(image_bgr, box, pad=args.crop_pad, min_size=args.min_crop_size)
            if crop is None:
                base = row.to_dict()
                base.update({"crop_valid": False})
                scored_rows.append(base)
                continue
            batch_crops.append(crop)
            batch_meta.append(row.to_dict())
            if len(batch_crops) >= args.batch_size:
                flush_batch()
    flush_batch()

    scored = pd.DataFrame(scored_rows)
    if "crop_valid" not in scored.columns:
        scored["crop_valid"] = False
    scored = add_detector_class_probability(scored, "clip")
    scored = add_detector_class_probability(scored, "yolo")
    scored["crop_valid"] = bool_series(scored["crop_valid"])
    scored["clip_same_as_detector"] = bool_series(scored["clip_same_as_detector"])
    scored["yolo_same_as_detector"] = bool_series(scored["yolo_same_as_detector"])
    timing = pd.DataFrame(timing_rows)

    scored.to_csv(scored_csv, index=False)
    timing.to_csv(timing_csv, index=False)
    if len(scored) <= 1_000_000:
        scored.to_excel(scored_xlsx, index=False)
    timing.to_excel(timing_xlsx, index=False)

    print(f"[{split}] Saved verifier scores: {scored_csv}")
    return scored, timing


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def build_gt_index(gt_df: pd.DataFrame):
    gt_index = {}
    for _, r in gt_df.reset_index(drop=True).iterrows():
        c = canonical_name(r["class_name"])
        if c not in WEAPON_CLASSES:
            continue
        key = (r["image_id"], c)
        gt_index.setdefault(key, []).append({
            "gt_local_idx": len(gt_index.get(key, [])),
            "box": np.array([r["x1"], r["y1"], r["x2"], r["y2"]], dtype=np.float32),
        })
    return gt_index


def greedy_fixed_metrics(det_df: pd.DataFrame, gt_df: pd.DataFrame, match_iou: float = 0.50) -> Dict[str, float]:
    gt_df = gt_df.copy()
    gt_df["class_name"] = gt_df["class_name"].map(canonical_name)
    gt_index = build_gt_index(gt_df)
    matched = {key: set() for key in gt_index.keys()}

    total_gt_by_class = {c: int((gt_df["class_name"] == c).sum()) for c in WEAPON_CLASSES}
    tp_by_class = {c: 0 for c in WEAPON_CLASSES}
    fp_by_class = {c: 0 for c in WEAPON_CLASSES}

    if len(det_df) > 0:
        det_sorted = det_df.sort_values("score", ascending=False)
    else:
        det_sorted = det_df

    for _, d in det_sorted.iterrows():
        c = canonical_name(d["class_name"])
        if c not in WEAPON_CLASSES:
            continue
        key = (d["image_id"], c)
        dbox = np.array([d["x1"], d["y1"], d["x2"], d["y2"]], dtype=np.float32)
        best_iou, best_j = 0.0, None
        for gt in gt_index.get(key, []):
            j = gt["gt_local_idx"]
            if j in matched.get(key, set()):
                continue
            val = iou_xyxy(dbox, gt["box"])
            if val > best_iou:
                best_iou, best_j = val, j
        if best_iou >= match_iou and best_j is not None:
            tp_by_class[c] += 1
            matched[key].add(best_j)
        else:
            fp_by_class[c] += 1

    fn_by_class = {c: total_gt_by_class[c] - tp_by_class[c] for c in WEAPON_CLASSES}
    tp = int(sum(tp_by_class.values()))
    fp = int(sum(fp_by_class.values()))
    fn = int(sum(fn_by_class.values()))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fdr = fp / (tp + fp) if (tp + fp) else 0.0

    out = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "custom_fp_ratio": fdr, "fdr": fdr,
        "num_predictions": int(tp + fp),
        "num_gt": int(tp + fn),
    }
    for c in WEAPON_CLASSES:
        ctp, cfp, cfn = tp_by_class[c], fp_by_class[c], fn_by_class[c]
        cp = ctp / (ctp + cfp) if (ctp + cfp) else 0.0
        cr = ctp / (ctp + cfn) if (ctp + cfn) else 0.0
        cf1 = 2 * cp * cr / (cp + cr) if (cp + cr) else 0.0
        out[f"{c}_tp"] = int(ctp)
        out[f"{c}_fp"] = int(cfp)
        out[f"{c}_fn"] = int(cfn)
        out[f"{c}_precision"] = cp
        out[f"{c}_recall"] = cr
        out[f"{c}_f1"] = cf1
    return out


def ap_for_class(det_df: pd.DataFrame, gt_df: pd.DataFrame, class_name: str, iou_thr: float) -> float:
    c = canonical_name(class_name)
    gt_c = gt_df[gt_df["class_name"].map(canonical_name) == c]
    n_gt = len(gt_c)
    if n_gt == 0:
        return np.nan
    gt_index = {}
    for _, r in gt_c.iterrows():
        gt_index.setdefault(r["image_id"], []).append({
            "box": np.array([r["x1"], r["y1"], r["x2"], r["y2"]], dtype=np.float32),
            "matched": False,
        })
    det_c = det_df[det_df["class_name"].map(canonical_name) == c].sort_values("score", ascending=False)
    if len(det_c) == 0:
        return 0.0
    tp = np.zeros(len(det_c), dtype=np.float32)
    fp = np.zeros(len(det_c), dtype=np.float32)
    for i, (_, d) in enumerate(det_c.iterrows()):
        image_id = d["image_id"]
        dbox = np.array([d["x1"], d["y1"], d["x2"], d["y2"]], dtype=np.float32)
        best_iou, best_idx = 0.0, None
        for j, gt in enumerate(gt_index.get(image_id, [])):
            if gt["matched"]:
                continue
            val = iou_xyxy(dbox, gt["box"])
            if val > best_iou:
                best_iou, best_idx = val, j
        if best_iou >= iou_thr and best_idx is not None:
            tp[i] = 1.0
            gt_index[image_id][best_idx]["matched"] = True
        else:
            fp[i] = 1.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(n_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, EPS)
    return average_precision_from_pr(recall, precision)


def compute_map(det_df: pd.DataFrame, gt_df: pd.DataFrame) -> Dict[str, float]:
    thresholds = np.arange(0.50, 0.96, 0.05)
    ap50s = []
    all_aps = []
    out = {}
    for c in WEAPON_CLASSES:
        ap50 = ap_for_class(det_df, gt_df, c, 0.50)
        out[f"ap50_{c}"] = ap50
        ap50s.append(ap50)
        cls_aps = [ap_for_class(det_df, gt_df, c, float(t)) for t in thresholds]
        out[f"ap50_95_{c}"] = float(np.nanmean(cls_aps))
        all_aps.extend(cls_aps)
    out["mAP50"] = float(np.nanmean(ap50s))
    out["mAP50_95"] = float(np.nanmean(all_aps))
    return out


# -----------------------------------------------------------------------------
# Decision rules
# -----------------------------------------------------------------------------

def suppress_duplicates_with_kept(df: pd.DataFrame, base_keep: pd.Series, recover_mask: pd.Series, duplicate_iou: float) -> pd.Series:
    if recover_mask.sum() == 0 or base_keep.sum() == 0:
        return recover_mask
    final_recover = recover_mask.copy()
    for (image_id, cls_name), g in df[recover_mask].groupby(["image_id", "class_name"], sort=False):
        kept = df[base_keep & (df["image_id"] == image_id) & (df["class_name"] == cls_name)]
        if len(kept) == 0:
            continue
        kept_boxes = kept[["x1", "y1", "x2", "y2"]].values.astype(np.float32)
        for idx, row in g.iterrows():
            box = row[["x1", "y1", "x2", "y2"]].values.astype(np.float32)
            if any(iou_xyxy(box, kb) >= duplicate_iou for kb in kept_boxes):
                final_recover.loc[idx] = False
    return final_recover


def make_decision(df: pd.DataFrame, cfg: Dict, duplicate_iou: float) -> Tuple[pd.Series, pd.Series, pd.Series, Optional[str]]:
    method = cfg["method"]
    verifier = cfg.get("verifier", "none")
    base_conf = float(cfg.get("base_conf", 0.25)) if not pd.isna(cfg.get("base_conf", 0.25)) else 0.25
    direct_conf = cfg.get("direct_conf", np.nan)
    verifier_prob_thr = cfg.get("verifier_prob_thr", np.nan)
    entropy_thr = cfg.get("entropy_thr", np.nan)
    recovery_min_conf = cfg.get("recovery_min_conf", np.nan)
    rescore_threshold = cfg.get("rescore_threshold", np.nan)
    rescore_alpha = cfg.get("rescore_alpha", np.nan)

    verified = pd.Series(False, index=df.index)
    recovered = pd.Series(False, index=df.index)
    score_override = None

    if method == "YOLO_threshold":
        keep = df["score"] >= base_conf
        return keep, verified, recovered, score_override

    if verifier not in ["clip", "yolo"]:
        raise ValueError(f"verifier must be clip/yolo for method {method}")

    valid_crop = df["crop_valid"].astype(bool)
    same = df[f"{verifier}_same_as_detector"].astype(bool)
    prob = df[f"{verifier}_prob_detector_class"].astype(float)
    ent = df[f"{verifier}_entropy"].astype(float)

    if method == "filter_hard":
        base = (df["score"] >= base_conf) & valid_crop
        keep = base & same
        verified = base
        return keep, verified, recovered, score_override

    if method == "filter_selective_conf":
        dc = float(direct_conf)
        vp = float(verifier_prob_thr)
        base = (df["score"] >= base_conf) & valid_crop
        direct = base & (df["score"] >= dc)
        need_verifier = base & (df["score"] < dc)
        keep = direct | (need_verifier & same & (prob >= vp))
        verified = need_verifier
        return keep, verified, recovered, score_override

    if method == "filter_selective_entropy":
        dc = float(direct_conf)
        eh = float(entropy_thr)
        base = (df["score"] >= base_conf) & valid_crop
        direct = base & (df["score"] >= dc)
        need_verifier = base & (df["score"] < dc)
        keep = direct | (need_verifier & same & (ent <= eh))
        verified = need_verifier
        return keep, verified, recovered, score_override

    if method == "recovery_conf":
        rmin = float(recovery_min_conf)
        vp = float(verifier_prob_thr)
        base_keep = df["score"] >= base_conf
        cand = (df["score"] >= rmin) & (df["score"] < base_conf) & valid_crop
        rec = cand & same & (prob >= vp)
        rec = suppress_duplicates_with_kept(df, base_keep, rec, duplicate_iou)
        keep = base_keep | rec
        verified = cand
        recovered = rec
        return keep, verified, recovered, score_override

    if method == "recovery_entropy":
        rmin = float(recovery_min_conf)
        eh = float(entropy_thr)
        base_keep = df["score"] >= base_conf
        cand = (df["score"] >= rmin) & (df["score"] < base_conf) & valid_crop
        rec = cand & same & (ent <= eh)
        rec = suppress_duplicates_with_kept(df, base_keep, rec, duplicate_iou)
        keep = base_keep | rec
        verified = cand
        recovered = rec
        return keep, verified, recovered, score_override

    if method == "filter_recovery_entropy":
        rmin = float(recovery_min_conf)
        dc = float(direct_conf)
        eh = float(entropy_thr)
        above = (df["score"] >= base_conf) & valid_crop
        direct = above & (df["score"] >= dc)
        need_filter = above & (df["score"] < dc)
        filtered_keep = direct | (need_filter & same & (ent <= eh))
        low_cand = (df["score"] >= rmin) & (df["score"] < base_conf) & valid_crop
        rec = low_cand & same & (ent <= eh)
        rec = suppress_duplicates_with_kept(df, filtered_keep, rec, duplicate_iou)
        keep = filtered_keep | rec
        verified = need_filter | low_cand
        recovered = rec
        return keep, verified, recovered, score_override

    if method == "semantic_rescore_product":
        new_score_col = f"_{verifier}_product_score"
        df[new_score_col] = df["score"].astype(float) * prob.fillna(0.0)
        keep = (df[new_score_col] >= float(rescore_threshold)) & valid_crop
        verified = (df["score"] >= float(recovery_min_conf)) & valid_crop
        score_override = new_score_col
        return keep, verified, recovered, score_override

    if method == "semantic_rescore_blend":
        alpha = float(rescore_alpha)
        new_score_col = f"_{verifier}_blend_score_{str(alpha).replace('.', 'p')}"
        df[new_score_col] = alpha * df["score"].astype(float) + (1.0 - alpha) * prob.fillna(0.0)
        keep = (df[new_score_col] >= float(rescore_threshold)) & valid_crop
        verified = (df["score"] >= float(recovery_min_conf)) & valid_crop
        score_override = new_score_col
        return keep, verified, recovered, score_override

    raise ValueError(f"Unknown method: {method}")


def method_label(cfg: Dict) -> str:
    method = cfg["method"]
    verifier = cfg.get("verifier", "none")
    prefix = "YOLO" if verifier == "none" else ("CLIP" if verifier == "clip" else "YOLOCLS")
    if method == "YOLO_threshold":
        return "YOLO_threshold_curve"
    if method == "filter_hard":
        return f"{prefix}_filter_hard"
    if method == "filter_selective_conf":
        return f"{prefix}_filter_selective_conf"
    if method == "filter_selective_entropy":
        return f"{prefix}_filter_selective_entropy"
    if method == "recovery_conf":
        return f"{prefix}_recovery_conf"
    if method == "recovery_entropy":
        return f"{prefix}_recovery_entropy"
    if method == "filter_recovery_entropy":
        return f"{prefix}_filter_recovery_entropy"
    if method == "semantic_rescore_product":
        return f"{prefix}_semantic_rescore_product"
    if method == "semantic_rescore_blend":
        return f"{prefix}_semantic_rescore_blend"
    return f"{prefix}_{method}"


def build_configs(detectors: List[str], args) -> List[Dict]:
    yolo_grid = parse_float_grid(args.yolo_threshold_grid)
    direct_grid = [x for x in parse_float_grid(args.direct_conf_grid) if x >= args.fixed_base_conf]
    prob_grid = parse_float_grid(args.verifier_prob_grid)
    ent_grid = parse_float_grid(args.entropy_grid)
    recovery_min_grid = [x for x in parse_float_grid(args.recovery_min_conf_grid) if x < args.fixed_base_conf]
    rescore_threshold_grid = parse_float_grid(args.rescore_threshold_grid)
    rescore_alpha_grid = parse_float_grid(args.rescore_alpha_grid)

    configs = []
    for det in detectors:
        for bc in yolo_grid:
            configs.append({
                "pair": det, "detector": det, "method": "YOLO_threshold", "verifier": "none",
                "base_conf": bc, "direct_conf": np.nan, "verifier_prob_thr": np.nan,
                "entropy_thr": np.nan, "recovery_min_conf": np.nan,
                "rescore_threshold": np.nan, "rescore_alpha": np.nan,
            })
        for verifier in ["clip", "yolo"]:
            configs.append({
                "pair": det, "detector": det, "method": "filter_hard", "verifier": verifier,
                "base_conf": args.fixed_base_conf, "direct_conf": np.nan,
                "verifier_prob_thr": np.nan, "entropy_thr": np.nan,
                "recovery_min_conf": np.nan, "rescore_threshold": np.nan, "rescore_alpha": np.nan,
            })
            for dc in direct_grid:
                for vp in prob_grid:
                    configs.append({
                        "pair": det, "detector": det, "method": "filter_selective_conf", "verifier": verifier,
                        "base_conf": args.fixed_base_conf, "direct_conf": dc,
                        "verifier_prob_thr": vp, "entropy_thr": np.nan,
                        "recovery_min_conf": np.nan, "rescore_threshold": np.nan, "rescore_alpha": np.nan,
                    })
                for eh in ent_grid:
                    configs.append({
                        "pair": det, "detector": det, "method": "filter_selective_entropy", "verifier": verifier,
                        "base_conf": args.fixed_base_conf, "direct_conf": dc,
                        "verifier_prob_thr": np.nan, "entropy_thr": eh,
                        "recovery_min_conf": np.nan, "rescore_threshold": np.nan, "rescore_alpha": np.nan,
                    })
            for rmin in recovery_min_grid:
                for vp in prob_grid:
                    configs.append({
                        "pair": det, "detector": det, "method": "recovery_conf", "verifier": verifier,
                        "base_conf": args.fixed_base_conf, "direct_conf": np.nan,
                        "verifier_prob_thr": vp, "entropy_thr": np.nan,
                        "recovery_min_conf": rmin, "rescore_threshold": np.nan, "rescore_alpha": np.nan,
                    })
                for eh in ent_grid:
                    configs.append({
                        "pair": det, "detector": det, "method": "recovery_entropy", "verifier": verifier,
                        "base_conf": args.fixed_base_conf, "direct_conf": np.nan,
                        "verifier_prob_thr": np.nan, "entropy_thr": eh,
                        "recovery_min_conf": rmin, "rescore_threshold": np.nan, "rescore_alpha": np.nan,
                    })
            for dc in direct_grid:
                for rmin in recovery_min_grid:
                    for eh in ent_grid:
                        configs.append({
                            "pair": det, "detector": det, "method": "filter_recovery_entropy", "verifier": verifier,
                            "base_conf": args.fixed_base_conf, "direct_conf": dc,
                            "verifier_prob_thr": np.nan, "entropy_thr": eh,
                            "recovery_min_conf": rmin, "rescore_threshold": np.nan, "rescore_alpha": np.nan,
                        })
            if not args.skip_rescore:
                for rmin in recovery_min_grid:
                    for rt in rescore_threshold_grid:
                        configs.append({
                            "pair": det, "detector": det, "method": "semantic_rescore_product", "verifier": verifier,
                            "base_conf": np.nan, "direct_conf": np.nan,
                            "verifier_prob_thr": np.nan, "entropy_thr": np.nan,
                            "recovery_min_conf": rmin, "rescore_threshold": rt, "rescore_alpha": np.nan,
                        })
                        for alpha in rescore_alpha_grid:
                            configs.append({
                                "pair": det, "detector": det, "method": "semantic_rescore_blend", "verifier": verifier,
                                "base_conf": np.nan, "direct_conf": np.nan,
                                "verifier_prob_thr": np.nan, "entropy_thr": np.nan,
                                "recovery_min_conf": rmin, "rescore_threshold": rt, "rescore_alpha": alpha,
                            })
    for c in configs:
        c["method_label"] = method_label(c)
    return configs


# -----------------------------------------------------------------------------
# Grid evaluation and summaries
# -----------------------------------------------------------------------------

def source_seconds(timing_df: pd.DataFrame) -> Dict[str, float]:
    if "total_sec_single_yolo" in timing_df.columns:
        col = "total_sec_single_yolo"
    elif "detector_inference_sec" in timing_df.columns:
        col = "detector_inference_sec"
    else:
        col = "total_sec_model1_model2_wbf"
    return timing_df.groupby("pair")[col].sum().to_dict()


def mean_verifier_seconds(scored_df: pd.DataFrame) -> Dict[str, float]:
    valid = scored_df["crop_valid"].astype(bool)
    clip_sec = float(scored_df.loc[valid, "clip_inference_sec_est"].mean()) if valid.any() else 0.0
    yolo_sec = float(scored_df.loc[valid, "yolo_inference_sec_est"].mean()) if valid.any() else 0.0
    return {"clip": clip_sec, "yolo": yolo_sec, "none": 0.0}


def evaluate_config(scored_df: pd.DataFrame, gt_df: pd.DataFrame, timing_df: pd.DataFrame,
                    cfg: Dict, split: str, args, compute_ap: bool = False) -> Dict:
    det_name = cfg["pair"]
    df = scored_df[scored_df["pair"] == det_name].copy()
    if cfg["method"].startswith("semantic_rescore"):
        rmin = float(cfg.get("recovery_min_conf", args.rescore_original_min_conf))
        df = df[df["score"] >= rmin].copy()

    keep, verified, recovered, score_override = make_decision(df, cfg, args.duplicate_iou)
    det = df[keep].copy()
    if score_override is not None and score_override in det.columns:
        det["original_detector_score"] = det["score"]
        det["score"] = det[score_override]

    metrics = greedy_fixed_metrics(det, gt_df, match_iou=args.match_iou)
    det_sec = float(source_seconds(timing_df).get(det_name, 0.0))
    v_sec_map = mean_verifier_seconds(scored_df)
    verifier = cfg.get("verifier", "none")
    n_verified = int(verified.sum())
    verifier_sec = n_verified * v_sec_map.get(verifier, 0.0)
    total_sec = det_sec + verifier_sec
    n_images = int(gt_df["image_id"].nunique())

    out = {
        "split": split,
        "pair": det_name,
        "detector": det_name,
        "method_label": cfg["method_label"],
        "method": cfg["method"],
        "verifier": verifier,
        "base_conf": cfg.get("base_conf", np.nan),
        "direct_conf": cfg.get("direct_conf", np.nan),
        "verifier_prob_thr": cfg.get("verifier_prob_thr", np.nan),
        "entropy_thr": cfg.get("entropy_thr", np.nan),
        "recovery_min_conf": cfg.get("recovery_min_conf", np.nan),
        "rescore_threshold": cfg.get("rescore_threshold", np.nan),
        "rescore_alpha": cfg.get("rescore_alpha", np.nan),
        "num_verified_candidates": n_verified,
        "num_recovered_candidates": int(recovered.sum()),
        "num_kept_detections": int(len(det)),
        "num_images": n_images,
        "single_yolo_inference_sec": det_sec,
        "verifier_sec_est": verifier_sec,
        "end_to_end_sec_est": total_sec,
        "end_to_end_images_per_sec_est": n_images / total_sec if total_sec > 0 else np.nan,
        **metrics,
    }
    if compute_ap:
        out.update(compute_map(det, gt_df))
    return out


def evaluate_grid(scored_df: pd.DataFrame, gt_df: pd.DataFrame, timing_df: pd.DataFrame,
                  configs: List[Dict], split: str, args) -> pd.DataFrame:
    rows = []
    for cfg in tqdm(configs, desc=f"{split}: Phase-6 config grid"):
        rows.append(evaluate_config(scored_df, gt_df, timing_df, cfg, split, args, compute_ap=False))
    return pd.DataFrame(rows)


def select_best_by_detector_method(grid_df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for (det, method_label), g in grid_df.groupby(["detector", "method_label"], sort=False):
        if args.selection_mode == "balanced":
            sort_cols = ["f1", "fp", "recall", "end_to_end_images_per_sec_est"]
            asc = [False, True, False, False]
        elif args.selection_mode == "fp_control":
            sort_cols = ["fp", "recall", "f1", "end_to_end_images_per_sec_est"]
            asc = [True, False, False, False]
        elif args.selection_mode == "fn_recovery":
            sort_cols = ["fn", "precision", "f1", "end_to_end_images_per_sec_est"]
            asc = [True, False, False, False]
        else:
            raise ValueError(f"Unknown selection_mode: {args.selection_mode}")
        rows.append(g.sort_values(sort_cols, ascending=asc).iloc[0].to_dict())
    return pd.DataFrame(rows)


def add_map_to_rows(row_df: pd.DataFrame, scored_df: pd.DataFrame, gt_df: pd.DataFrame,
                    timing_df: pd.DataFrame, split: str, args, desc: str) -> pd.DataFrame:
    rows = []
    keys = ["pair", "detector", "method", "method_label", "verifier", "base_conf", "direct_conf",
            "verifier_prob_thr", "entropy_thr", "recovery_min_conf", "rescore_threshold", "rescore_alpha"]
    for _, r in tqdm(row_df.iterrows(), total=len(row_df), desc=desc):
        cfg = {k: r[k] for k in keys if k in r.index}
        if "pair" not in cfg and "detector" in cfg:
            cfg["pair"] = cfg["detector"]
        rows.append(evaluate_config(scored_df, gt_df, timing_df, cfg, split, args, compute_ap=True))
    return pd.DataFrame(rows)


def apply_valid_selected_to_test(valid_best: pd.DataFrame, test_scored: pd.DataFrame,
                                 test_gt: pd.DataFrame, test_timing: pd.DataFrame, args) -> pd.DataFrame:
    return add_map_to_rows(valid_best, test_scored, test_gt, test_timing, "test", args, "test: frozen Phase-6 configs")


def mean_std_table(df: pd.DataFrame, group_col: str = "method_label") -> pd.DataFrame:
    cols = [
        "precision", "recall", "f1", "custom_fp_ratio", "fdr",
        "tp", "fp", "fn", "mAP50", "mAP50_95",
        "end_to_end_images_per_sec_est", "num_verified_candidates", "num_recovered_candidates", "num_kept_detections",
    ]
    rows = []
    for key, g in df.groupby(group_col, sort=False):
        row = {group_col: key, "num_detectors": int(g["detector"].nunique())}
        for c in cols:
            if c in g.columns:
                row[f"{c}_mean"] = float(g[c].mean())
                row[f"{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_curve(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["method_label"] == "YOLO_threshold_curve"].copy()


def matched_recall_table(method_df: pd.DataFrame, yolo_curve: pd.DataFrame, tolerance: float, split: str) -> pd.DataFrame:
    rows = []
    for _, r in method_df.iterrows():
        if r["method_label"] == "YOLO_threshold_curve":
            continue
        g = yolo_curve[yolo_curve["detector"] == r["detector"]].copy()
        if len(g) == 0:
            continue
        candidates = g[g["recall"] >= (float(r["recall"]) - tolerance)].copy()
        matched_type = "recall_at_or_above"
        if len(candidates) == 0:
            candidates = g.copy()
            candidates["recall_distance"] = (candidates["recall"] - float(r["recall"])).abs()
            candidates = candidates.sort_values(["recall_distance", "fp", "f1"], ascending=[True, True, False])
            matched_type = "closest_recall_no_yolo_at_or_above"
        else:
            candidates = candidates.sort_values(["fp", "recall", "f1"], ascending=[True, False, False])
        base = candidates.iloc[0]
        rows.append({
            "split": split,
            "detector": r["detector"],
            "method_label": r["method_label"],
            "method_precision": r["precision"],
            "method_recall": r["recall"],
            "method_f1": r["f1"],
            "method_fp": int(r["fp"]),
            "method_fn": int(r["fn"]),
            "single_yolo_matched_conf": base["base_conf"],
            "single_yolo_precision": base["precision"],
            "single_yolo_recall": base["recall"],
            "single_yolo_f1": base["f1"],
            "single_yolo_fp": int(base["fp"]),
            "single_yolo_fn": int(base["fn"]),
            "fp_change_vs_single_yolo": int(r["fp"]) - int(base["fp"]),
            "fn_change_vs_single_yolo": int(r["fn"]) - int(base["fn"]),
            "f1_change_vs_single_yolo": float(r["f1"]) - float(base["f1"]),
            "recall_difference": float(r["recall"]) - float(base["recall"]),
            "matched_type": matched_type,
        })
    return pd.DataFrame(rows)


def matched_precision_table(method_df: pd.DataFrame, yolo_curve: pd.DataFrame, tolerance: float, split: str) -> pd.DataFrame:
    rows = []
    for _, r in method_df.iterrows():
        if r["method_label"] == "YOLO_threshold_curve":
            continue
        g = yolo_curve[yolo_curve["detector"] == r["detector"]].copy()
        if len(g) == 0:
            continue
        candidates = g[g["precision"] >= (float(r["precision"]) - tolerance)].copy()
        matched_type = "precision_at_or_above"
        if len(candidates) == 0:
            candidates = g.copy()
            candidates["precision_distance"] = (candidates["precision"] - float(r["precision"])).abs()
            candidates = candidates.sort_values(["precision_distance", "fn", "f1"], ascending=[True, True, False])
            matched_type = "closest_precision_no_yolo_at_or_above"
        else:
            candidates = candidates.sort_values(["fn", "precision", "f1"], ascending=[True, False, False])
        base = candidates.iloc[0]
        rows.append({
            "split": split,
            "detector": r["detector"],
            "method_label": r["method_label"],
            "method_precision": r["precision"],
            "method_recall": r["recall"],
            "method_f1": r["f1"],
            "method_fp": int(r["fp"]),
            "method_fn": int(r["fn"]),
            "single_yolo_matched_conf": base["base_conf"],
            "single_yolo_precision": base["precision"],
            "single_yolo_recall": base["recall"],
            "single_yolo_f1": base["f1"],
            "single_yolo_fp": int(base["fp"]),
            "single_yolo_fn": int(base["fn"]),
            "fp_change_vs_single_yolo": int(r["fp"]) - int(base["fp"]),
            "fn_change_vs_single_yolo": int(r["fn"]) - int(base["fn"]),
            "f1_change_vs_single_yolo": float(r["f1"]) - float(base["f1"]),
            "precision_difference": float(r["precision"]) - float(base["precision"]),
            "matched_type": matched_type,
        })
    return pd.DataFrame(rows)


def candidate_final_configs(valid_best: pd.DataFrame, args) -> pd.DataFrame:
    g = valid_best[valid_best["verifier"].isin(["clip", "yolo"])].copy()
    if len(g) == 0:
        return g
    if args.final_candidate_mode == "balanced":
        sort_cols = ["f1", "fp", "recall", "end_to_end_images_per_sec_est"]
        asc = [False, True, False, False]
    elif args.final_candidate_mode == "fp_control":
        sort_cols = ["fp", "recall", "f1", "end_to_end_images_per_sec_est"]
        asc = [True, False, False, False]
    elif args.final_candidate_mode == "fn_recovery":
        sort_cols = ["fn", "precision", "f1", "end_to_end_images_per_sec_est"]
        asc = [True, False, False, False]
    else:
        raise ValueError(f"Unknown final_candidate_mode: {args.final_candidate_mode}")
    return g.sort_values(sort_cols, ascending=asc).head(args.top_k_final_candidates).reset_index(drop=True)


def clean_scored_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["class_name"] = df["class_name"].map(canonical_name)
    df = df[df["class_name"].isin(WEAPON_CLASSES)].reset_index(drop=True)
    df["crop_valid"] = bool_series(df["crop_valid"])
    df["clip_same_as_detector"] = bool_series(df["clip_same_as_detector"])
    df["yolo_same_as_detector"] = bool_series(df["yolo_same_as_detector"])
    num_cols = [
        "score", "x1", "y1", "x2", "y2",
        "clip_prob_detector_class", "clip_entropy", "clip_inference_sec_est",
        "yolo_prob_detector_class", "yolo_entropy", "yolo_inference_sec_est",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["score", "x1", "y1", "x2", "y2"]).reset_index(drop=True)
    return df


def clean_gt_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["class_name"] = df["class_name"].map(canonical_name)
    df = df[df["class_name"].isin(WEAPON_CLASSES)].reset_index(drop=True)
    for c in ["x1", "y1", "x2", "y2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["x1", "y1", "x2", "y2"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6 single-YOLO verifier control experiment")

    p.add_argument("--data_yaml", type=str, default="/home/inspur/ishrai/yolo_server.yaml")
    p.add_argument("--model_dir", type=str, default="/home/inspur/ishrai/all_togather")
    p.add_argument("--detectors", type=str, default="y8x,y8l,y26n,y26s",
                   help="Comma-separated detector keys: y8x,y8l,y26n,y26s")
    p.add_argument("--out_dir", type=str, default="/home/inspur/ishrai/phase6_single_yolo_all_detectors")

    p.add_argument("--clip_ckpt", type=str, default="/home/inspur/ishrai/all_togather/clip_verifier/best_clip_finetuned.pth")
    p.add_argument("--clip_model_name", type=str, default="ViT-B-32")
    p.add_argument("--clip_pretrained", type=str,
                   default="/home/inspur/ishrai/pretrained_openclip/ViT-B-32_laion2b_e16/open_clip_model.safetensors")
    p.add_argument("--yolo_cls_ckpt", type=str, default="/home/inspur/ishrai/all_togather/yolo_verifier/YOLOv8l_cls_best.pt")
    p.add_argument("--yolo_cls_imgsz", type=int, default=224)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--low_conf", type=float, default=0.001)
    p.add_argument("--nms_iou", type=float, default=0.70)
    p.add_argument("--max_det", type=int, default=300)
    p.add_argument("--save_conf", type=float, default=0.01,
                   help="Save candidates at or above this confidence. Must be low enough for recovery analysis.")
    p.add_argument("--fixed_base_conf", type=float, default=0.25,
                   help="Reference single-YOLO threshold for filter/recovery modes.")
    p.add_argument("--match_iou", type=float, default=0.50)
    p.add_argument("--duplicate_iou", type=float, default=0.50)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--crop_pad", type=float, default=0.05)
    p.add_argument("--min_crop_size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Optional image limit for debug runs. 0 means full split.")

    p.add_argument("--yolo_threshold_grid", type=str,
                   default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90")
    p.add_argument("--direct_conf_grid", type=str, default="0.50,0.60,0.70,0.80,0.90")
    p.add_argument("--verifier_prob_grid", type=str, default="0.50,0.60,0.70,0.80,0.90")
    p.add_argument("--entropy_grid", type=str, default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    p.add_argument("--recovery_min_conf_grid", type=str, default="0.05,0.10,0.15,0.20")
    p.add_argument("--rescore_threshold_grid", type=str,
                   default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90")
    p.add_argument("--rescore_alpha_grid", type=str, default="0.50,0.70")
    p.add_argument("--rescore_original_min_conf", type=float, default=0.05)
    p.add_argument("--skip_rescore", action="store_true", help="Skip semantic-rescore methods for a faster control experiment.")

    p.add_argument("--matched_tolerance", type=float, default=0.002)
    p.add_argument("--selection_mode", type=str, default="balanced", choices=["balanced", "fp_control", "fn_recovery"])
    p.add_argument("--final_candidate_mode", type=str, default="balanced", choices=["balanced", "fp_control", "fn_recovery"])
    p.add_argument("--top_k_final_candidates", type=int, default=20)

    p.add_argument("--reuse_candidates", action="store_true", help="Reuse saved single-YOLO candidate CSVs if present.")
    p.add_argument("--reuse_scores", action="store_true", help="Reuse saved verifier-score CSVs if present.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    detectors = parse_list(args.detectors)
    if not detectors:
        raise RuntimeError("No detectors selected. Use --detectors y8x,y8l,y26n,y26s")

    settings = vars(args).copy()
    settings.update({
        "note": "Phase 6 tests single YOLO detectors with the same CLIP/YOLO-cls verifier logic used for WBF.",
        "protocol": "Validation selects configurations; test applies validation-selected configurations only.",
        "detectors_evaluated": detectors,
    })
    safe_json_dump(settings, out_dir / "phase6_single_yolo_settings.json")

    # Step 1: candidate generation on valid and test.
    valid_cand, valid_gt, valid_timing = generate_candidates_for_split("valid", args, out_dir)
    test_cand, test_gt, test_timing = generate_candidates_for_split("test", args, out_dir)

    # Step 2: verifier scoring on valid and test.
    valid_scored, valid_ver_timing = run_verifier_scoring_for_split("valid", valid_cand, args, out_dir)
    test_scored, test_ver_timing = run_verifier_scoring_for_split("test", test_cand, args, out_dir)

    valid_scored = clean_scored_dataframe(valid_scored)
    test_scored = clean_scored_dataframe(test_scored)
    valid_gt = clean_gt_dataframe(valid_gt)
    test_gt = clean_gt_dataframe(test_gt)

    common_detectors = sorted(set(valid_scored["pair"].dropna().unique()).intersection(set(test_scored["pair"].dropna().unique())))
    common_detectors = [d for d in common_detectors if d in detectors]
    if not common_detectors:
        raise RuntimeError("No common detectors found between validation and test scored files.")

    # Step 3: validation grid and best config selection.
    configs = build_configs(common_detectors, args)
    print("=" * 80)
    print("Phase 6 evaluation")
    print(f"Detectors: {common_detectors}")
    print(f"Validation candidates: {len(valid_scored)}")
    print(f"Test candidates      : {len(test_scored)}")
    print(f"Validation GT boxes  : {len(valid_gt)}")
    print(f"Test GT boxes        : {len(test_gt)}")
    print(f"Total validation configs: {len(configs)}")
    print("=" * 80)

    valid_grid = evaluate_grid(valid_scored, valid_gt, valid_timing, configs, "valid", args)
    valid_grid_csv = out_dir / "phase6_valid_full_grid.csv"
    valid_grid_xlsx = out_dir / "phase6_valid_full_grid.xlsx"
    valid_grid.to_csv(valid_grid_csv, index=False)
    valid_grid.to_excel(valid_grid_xlsx, index=False)

    valid_best_no_map = select_best_by_detector_method(valid_grid, args)
    valid_best = add_map_to_rows(valid_best_no_map, valid_scored, valid_gt, valid_timing, "valid", args,
                                 "valid: AP for best Phase-6 configs")
    valid_best_csv = out_dir / "phase6_valid_best_by_detector_method.csv"
    valid_best_xlsx = out_dir / "phase6_valid_best_by_detector_method.xlsx"
    valid_best.to_csv(valid_best_csv, index=False)
    valid_best.to_excel(valid_best_xlsx, index=False)

    # Step 4: apply validation-selected settings to test.
    test_frozen = apply_valid_selected_to_test(valid_best, test_scored, test_gt, test_timing, args)
    test_frozen_csv = out_dir / "phase6_test_frozen_results.csv"
    test_frozen_xlsx = out_dir / "phase6_test_frozen_results.xlsx"
    test_frozen.to_csv(test_frozen_csv, index=False)
    test_frozen.to_excel(test_frozen_xlsx, index=False)

    # Step 5: summaries and matched operating-point diagnostics.
    valid_mean_std = mean_std_table(valid_best)
    test_mean_std = mean_std_table(test_frozen)
    valid_mean_std.to_csv(out_dir / "phase6_valid_mean_std_across_detectors.csv", index=False)
    test_mean_std.to_csv(out_dir / "phase6_test_mean_std_across_detectors.csv", index=False)
    valid_mean_std.to_excel(out_dir / "phase6_valid_mean_std_across_detectors.xlsx", index=False)
    test_mean_std.to_excel(out_dir / "phase6_test_mean_std_across_detectors.xlsx", index=False)

    valid_curve = threshold_curve(valid_grid)
    test_curve = threshold_curve(test_frozen)
    # For a richer test threshold curve, evaluate only YOLO threshold configs on test too.
    yolo_only_test_configs = [c for c in configs if c["method_label"] == "YOLO_threshold_curve"]
    test_curve_full = evaluate_grid(test_scored, test_gt, test_timing, yolo_only_test_configs, "test", args)
    test_curve_full = add_map_to_rows(test_curve_full, test_scored, test_gt, test_timing, "test", args,
                                      "test: AP for single-YOLO threshold curve")
    valid_curve.to_csv(out_dir / "phase6_valid_single_yolo_threshold_curve.csv", index=False)
    test_curve_full.to_csv(out_dir / "phase6_test_single_yolo_threshold_curve.csv", index=False)
    valid_curve.to_excel(out_dir / "phase6_valid_single_yolo_threshold_curve.xlsx", index=False)
    test_curve_full.to_excel(out_dir / "phase6_test_single_yolo_threshold_curve.xlsx", index=False)

    valid_match_recall = matched_recall_table(valid_best, valid_curve, args.matched_tolerance, "valid")
    valid_match_precision = matched_precision_table(valid_best, valid_curve, args.matched_tolerance, "valid")
    test_match_recall = matched_recall_table(test_frozen, test_curve_full, args.matched_tolerance, "test")
    test_match_precision = matched_precision_table(test_frozen, test_curve_full, args.matched_tolerance, "test")
    for name, df in [
        ("phase6_valid_matched_recall_vs_single_yolo", valid_match_recall),
        ("phase6_valid_matched_precision_vs_single_yolo", valid_match_precision),
        ("phase6_test_matched_recall_vs_single_yolo", test_match_recall),
        ("phase6_test_matched_precision_vs_single_yolo", test_match_precision),
    ]:
        df.to_csv(out_dir / f"{name}.csv", index=False)
        df.to_excel(out_dir / f"{name}.xlsx", index=False)

    top_valid = candidate_final_configs(valid_best, args)
    top_valid_on_test = test_frozen.merge(
        top_valid[["detector", "method_label", "verifier", "base_conf", "direct_conf", "verifier_prob_thr", "entropy_thr", "recovery_min_conf", "rescore_threshold", "rescore_alpha"]],
        on=["detector", "method_label", "verifier", "base_conf", "direct_conf", "verifier_prob_thr", "entropy_thr", "recovery_min_conf", "rescore_threshold", "rescore_alpha"],
        how="inner",
    ) if len(top_valid) else pd.DataFrame()
    top_valid.to_csv(out_dir / "phase6_top_validation_final_candidates.csv", index=False)
    top_valid.to_excel(out_dir / "phase6_top_validation_final_candidates.xlsx", index=False)
    top_valid_on_test.to_csv(out_dir / "phase6_top_validation_candidates_on_test.csv", index=False)
    top_valid_on_test.to_excel(out_dir / "phase6_top_validation_candidates_on_test.xlsx", index=False)

    safe_write_excel(out_dir / "phase6_single_yolo_paper_summary.xlsx", {
        "top_validation_candidates": top_valid,
        "top_candidates_on_test": top_valid_on_test,
        "valid_best_by_method": valid_best,
        "test_frozen_results": test_frozen,
        "valid_mean_std": valid_mean_std,
        "test_mean_std": test_mean_std,
        "valid_matched_recall": valid_match_recall,
        "test_matched_recall": test_match_recall,
        "valid_matched_precision": valid_match_precision,
        "test_matched_precision": test_match_precision,
    })

    print("\n" + "=" * 80)
    print("PHASE 6 SINGLE-YOLO CONTROL COMPLETE")
    print(f"Output folder                       : {out_dir}")
    print(f"Validation full grid                : {valid_grid_xlsx}")
    print(f"Validation best by detector/method  : {valid_best_xlsx}")
    print(f"Test frozen results                 : {test_frozen_xlsx}")
    print(f"Paper summary workbook              : {out_dir / 'phase6_single_yolo_paper_summary.xlsx'}")
    print("\nTop validation candidates:")
    show_cols = ["detector", "method_label", "precision", "recall", "f1", "fp", "fn", "base_conf", "direct_conf", "verifier_prob_thr", "entropy_thr", "recovery_min_conf", "mAP50", "mAP50_95", "end_to_end_images_per_sec_est"]
    show_cols = [c for c in show_cols if c in top_valid.columns]
    if len(top_valid):
        print(top_valid[show_cols].head(20).to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
