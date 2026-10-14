#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 10: HN-WBF + CLIP / YOLO-cls verifier final analysis.

Purpose
-------
This is the final evaluation phase before selecting the paper direction.
It uses the HN-fine-tuned YOLO/WBF candidates generated in Phase 9 and tests
whether CLIP or YOLO-cls verification can recover the recall loss introduced
by hard-negative fine-tuning, while controlling false positives.

What it does
------------
1) Loads Phase-9 HN-WBF candidates for validation and test.
2) Scores each candidate crop using:
   - fine-tuned CLIP verifier
   - fine-tuned YOLOv8l-cls verifier
3) Evaluates validation threshold grids for:
   - HN-WBF threshold tuning
   - filter-only verification
   - recovery-only verification
   - filter+recovery verification
   - optional semantic rescoring
4) Selects thresholds on validation only.
5) Applies frozen validation-selected settings to test.
6) Saves CSV/XLSX outputs for paper analysis.

Important policy
----------------
Validation selects configurations. Test is used only for frozen evaluation.
This script does not train anything.
This script does not rerun HN-YOLO detection or WBF if Phase-9 candidate files exist.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch

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

WEAPON_CLASSES = ["knife", "pistol"]
UNIFIED_CLASSES = ["knife", "pistol", "non_weapon"]
EPS = 1e-12


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_json_dump(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_float_grid(text: str) -> List[float]:
    vals = []
    for part in str(text).split(','):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def canonical_name(x) -> str:
    s = str(x).strip().lower()
    s = s.replace('-', '_').replace(' ', '_')
    if s in ["background", "nonweapon", "non_weapon_object", "non_weapon_objects", "negative", "bg"]:
        return "non_weapon"
    return s


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


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


# -----------------------------------------------------------------------------
# HN Phase-9 loading and standardization
# -----------------------------------------------------------------------------

def standardize_hn_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pair" not in df.columns:
        if "entity" in df.columns:
            df["pair"] = df["entity"].astype(str)
        elif "detector" in df.columns:
            df["pair"] = df["detector"].astype(str)
        else:
            raise RuntimeError("Cannot find pair/entity/detector column in candidate CSV.")
    if "score" not in df.columns:
        if "conf" in df.columns:
            df["score"] = pd.to_numeric(df["conf"], errors="coerce")
        else:
            raise RuntimeError("Cannot find score/conf column in candidate CSV.")
    required = ["pair", "image_id", "image_path", "class_name", "score", "x1", "y1", "x2", "y2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Candidate CSV missing required columns: {missing}")
    df["class_name"] = df["class_name"].map(canonical_name)
    df = df[df["class_name"].isin(WEAPON_CLASSES)].reset_index(drop=True)
    for c in ["score", "x1", "y1", "x2", "y2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["score", "x1", "y1", "x2", "y2"]).reset_index(drop=True)
    return df


def standardize_gt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ["image_id", "class_name", "x1", "y1", "x2", "y2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Ground truth CSV missing required columns: {missing}")
    df["class_name"] = df["class_name"].map(canonical_name)
    df = df[df["class_name"].isin(WEAPON_CLASSES)].reset_index(drop=True)
    for c in ["x1", "y1", "x2", "y2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["x1", "y1", "x2", "y2"]).reset_index(drop=True)


def standardize_timing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pair" not in df.columns:
        if "entity" in df.columns:
            df["pair"] = df["entity"].astype(str)
        else:
            raise RuntimeError("Timing CSV missing pair/entity column.")
    if "total_sec_model1_model2_wbf" not in df.columns:
        if "total_pipeline_sec_est" in df.columns:
            df["total_sec_model1_model2_wbf"] = pd.to_numeric(df["total_pipeline_sec_est"], errors="coerce")
        elif "images_per_sec_est" in df.columns and "n_images" in df.columns:
            df["total_sec_model1_model2_wbf"] = pd.to_numeric(df["n_images"], errors="coerce") / pd.to_numeric(df["images_per_sec_est"], errors="coerce")
        else:
            raise RuntimeError("Timing CSV missing total_pipeline_sec_est or compatible timing columns.")
    return df


def load_scored_split(scores_path: Path, gt_path: Path, timing_path: Path, split_name: str):
    for p in [scores_path, gt_path, timing_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required {split_name} file: {p}")
    cand = pd.read_csv(scores_path)
    gt = pd.read_csv(gt_path)
    timing = pd.read_csv(timing_path)
    cand = standardize_hn_candidates(cand)
    gt = standardize_gt(gt)
    timing = standardize_timing(timing)
    required_score_cols = [
        "crop_valid", "clip_prob_detector_class", "clip_same_as_detector", "clip_entropy", "clip_inference_sec_est",
        "yolo_prob_detector_class", "yolo_same_as_detector", "yolo_entropy", "yolo_inference_sec_est",
    ]
    missing = [c for c in required_score_cols if c not in cand.columns]
    if missing:
        raise RuntimeError(f"Scored candidate file is missing verifier columns: {missing}\nFile: {scores_path}")
    cand["crop_valid"] = bool_series(cand["crop_valid"])
    cand["clip_same_as_detector"] = bool_series(cand["clip_same_as_detector"])
    cand["yolo_same_as_detector"] = bool_series(cand["yolo_same_as_detector"])
    for c in ["clip_prob_detector_class", "clip_entropy", "clip_inference_sec_est", "yolo_prob_detector_class", "yolo_entropy", "yolo_inference_sec_est"]:
        cand[c] = pd.to_numeric(cand[c], errors="coerce")
    return cand, gt, timing


# -----------------------------------------------------------------------------
# Crop and verifier scoring
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
    model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model_name, pretrained=pretrained_arg, device=device)
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
        raise RuntimeError(f"YOLO-cls names cannot be mapped to {UNIFIED_CLASSES}. Missing: {missing}. names={names}")
    return model, canon_to_idx


def score_yolo_cls_batch(model, canon_to_idx: Dict[str, int], crops: List[Image.Image], args) -> np.ndarray:
    sources = [np.asarray(im.convert("RGB")) for im in crops]
    results = model.predict(source=sources, imgsz=args.yolo_cls_imgsz, device=args.device, verbose=False, batch=len(sources))
    out = []
    for r in results:
        if r.probs is None:
            raise RuntimeError("YOLO classification result has no r.probs. Check that checkpoint is a classification model.")
        p = r.probs.data.detach().cpu().numpy().astype(np.float32)
        aligned = np.zeros(3, dtype=np.float32)
        aligned[0] = p[canon_to_idx["knife"]]
        aligned[1] = p[canon_to_idx["pistol"]]
        aligned[2] = p[canon_to_idx["non_weapon"]]
        out.append(aligned)
    return np.vstack(out).astype(np.float32)


def run_verifier_scoring(cand_df: pd.DataFrame, args, out_dir: Path, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    clip_model, clip_preprocess, clip_text_features = load_clip_verifier(args, device)
    yolo_cls_model, yolo_cls_map = load_yolo_cls(args)

    scored_rows = []
    timing_rows = []
    total = len(cand_df)
    print(f"Scoring {total} HN-WBF candidates for split={split}.")

    grouped = cand_df.groupby("image_path", sort=False)
    batch_crops, batch_meta = [], []

    def add_invalid(row_dict):
        base = dict(row_dict)
        base.update({
            "crop_valid": False,
            "clip_prob_knife": np.nan, "clip_prob_pistol": np.nan, "clip_prob_non_weapon": np.nan,
            "clip_pred_id": -1, "clip_pred_name": "invalid_crop", "clip_entropy": np.nan,
            "clip_inference_sec_est": 0.0,
            "yolo_prob_knife": np.nan, "yolo_prob_pistol": np.nan, "yolo_prob_non_weapon": np.nan,
            "yolo_pred_id": -1, "yolo_pred_name": "invalid_crop", "yolo_entropy": np.nan,
            "yolo_inference_sec_est": 0.0,
        })
        scored_rows.append(base)

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

    for image_path, g in tqdm(grouped, total=cand_df["image_path"].nunique(), desc=f"{split}: images"):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"WARNING: cannot read image: {image_path}")
            for _, row in g.iterrows():
                add_invalid(row.to_dict())
            continue
        h, w = image_bgr.shape[:2]
        for _, row in g.iterrows():
            box = maybe_abs_to_norm_box(row, w, h)
            crop = crop_candidate(image_bgr, box, pad=args.crop_pad, min_size=args.min_crop_size)
            if crop is None:
                add_invalid(row.to_dict())
                continue
            batch_crops.append(crop)
            batch_meta.append(row.to_dict())
            if len(batch_crops) >= args.batch_size:
                flush_batch()
    flush_batch()

    scored = pd.DataFrame(scored_rows)
    scored = add_detector_class_probability(scored, "clip")
    scored = add_detector_class_probability(scored, "yolo")
    timing = pd.DataFrame(timing_rows)

    scored_csv = out_dir / f"phase10_{split}_HN_candidate_verifier_scores.csv"
    scored_xlsx = out_dir / f"phase10_{split}_HN_candidate_verifier_scores.xlsx"
    timing_csv = out_dir / f"phase10_{split}_HN_verifier_batch_timing.csv"
    timing_xlsx = out_dir / f"phase10_{split}_HN_verifier_batch_timing.xlsx"
    scored.to_csv(scored_csv, index=False)
    scored.to_excel(scored_xlsx, index=False)
    timing.to_csv(timing_csv, index=False)
    timing.to_excel(timing_xlsx, index=False)
    print(f"Saved candidate verifier scores: {scored_csv}")
    print(f"Saved candidate verifier scores Excel: {scored_xlsx}")
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
    gt_index = build_gt_index(gt_df)
    matched = {key: set() for key in gt_index.keys()}
    total_gt_by_class = {c: int((gt_df["class_name"].map(canonical_name) == c).sum()) for c in WEAPON_CLASSES}
    tp_by_class = {c: 0 for c in WEAPON_CLASSES}
    fp_by_class = {c: 0 for c in WEAPON_CLASSES}
    det_sorted = det_df.sort_values("score", ascending=False) if len(det_df) > 0 else det_df
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
    out = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
           "custom_fp_ratio": fdr, "fdr": fdr, "num_predictions": int(tp + fp), "num_gt": int(tp + fn)}
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
        cls_aps = []
        for t in thresholds:
            cls_aps.append(ap_for_class(det_df, gt_df, c, float(t)))
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

    if method == "WBF_threshold":
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
    prefix = "HN_WBF" if verifier == "none" else ("HN_CLIP" if verifier == "clip" else "HN_YOLO")
    if method == "WBF_threshold":
        return "HN_WBF_threshold_curve"
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


def build_configs(pairs: List[str], args) -> List[Dict]:
    wbf_grid = parse_float_grid(args.wbf_threshold_grid)
    direct_grid = [x for x in parse_float_grid(args.direct_conf_grid) if x >= args.fixed_base_conf]
    prob_grid = parse_float_grid(args.verifier_prob_grid)
    ent_grid = parse_float_grid(args.entropy_grid)
    recovery_min_grid = [x for x in parse_float_grid(args.recovery_min_conf_grid) if x < args.fixed_base_conf]
    rescore_threshold_grid = parse_float_grid(args.rescore_threshold_grid)
    rescore_alpha_grid = parse_float_grid(args.rescore_alpha_grid)
    configs = []
    for pair in pairs:
        for bc in wbf_grid:
            configs.append({"pair": pair, "method": "WBF_threshold", "verifier": "none", "base_conf": bc,
                            "direct_conf": np.nan, "verifier_prob_thr": np.nan, "entropy_thr": np.nan,
                            "recovery_min_conf": np.nan, "rescore_threshold": np.nan, "rescore_alpha": np.nan})
        for verifier in ["clip", "yolo"]:
            configs.append({"pair": pair, "method": "filter_hard", "verifier": verifier, "base_conf": args.fixed_base_conf,
                            "direct_conf": np.nan, "verifier_prob_thr": np.nan, "entropy_thr": np.nan,
                            "recovery_min_conf": np.nan, "rescore_threshold": np.nan, "rescore_alpha": np.nan})
            for dc in direct_grid:
                for vp in prob_grid:
                    configs.append({"pair": pair, "method": "filter_selective_conf", "verifier": verifier,
                                    "base_conf": args.fixed_base_conf, "direct_conf": dc, "verifier_prob_thr": vp,
                                    "entropy_thr": np.nan, "recovery_min_conf": np.nan, "rescore_threshold": np.nan,
                                    "rescore_alpha": np.nan})
                for eh in ent_grid:
                    configs.append({"pair": pair, "method": "filter_selective_entropy", "verifier": verifier,
                                    "base_conf": args.fixed_base_conf, "direct_conf": dc, "verifier_prob_thr": np.nan,
                                    "entropy_thr": eh, "recovery_min_conf": np.nan, "rescore_threshold": np.nan,
                                    "rescore_alpha": np.nan})
            for rmin in recovery_min_grid:
                for vp in prob_grid:
                    configs.append({"pair": pair, "method": "recovery_conf", "verifier": verifier,
                                    "base_conf": args.fixed_base_conf, "direct_conf": np.nan, "verifier_prob_thr": vp,
                                    "entropy_thr": np.nan, "recovery_min_conf": rmin, "rescore_threshold": np.nan,
                                    "rescore_alpha": np.nan})
                for eh in ent_grid:
                    configs.append({"pair": pair, "method": "recovery_entropy", "verifier": verifier,
                                    "base_conf": args.fixed_base_conf, "direct_conf": np.nan, "verifier_prob_thr": np.nan,
                                    "entropy_thr": eh, "recovery_min_conf": rmin, "rescore_threshold": np.nan,
                                    "rescore_alpha": np.nan})
            for dc in direct_grid:
                for rmin in recovery_min_grid:
                    for eh in ent_grid:
                        configs.append({"pair": pair, "method": "filter_recovery_entropy", "verifier": verifier,
                                        "base_conf": args.fixed_base_conf, "direct_conf": dc,
                                        "verifier_prob_thr": np.nan, "entropy_thr": eh, "recovery_min_conf": rmin,
                                        "rescore_threshold": np.nan, "rescore_alpha": np.nan})
            if not args.skip_rescore:
                for rmin in recovery_min_grid:
                    for rt in rescore_threshold_grid:
                        configs.append({"pair": pair, "method": "semantic_rescore_product", "verifier": verifier,
                                        "base_conf": np.nan, "direct_conf": np.nan, "verifier_prob_thr": np.nan,
                                        "entropy_thr": np.nan, "recovery_min_conf": rmin, "rescore_threshold": rt,
                                        "rescore_alpha": np.nan})
                        for alpha in rescore_alpha_grid:
                            configs.append({"pair": pair, "method": "semantic_rescore_blend", "verifier": verifier,
                                            "base_conf": np.nan, "direct_conf": np.nan, "verifier_prob_thr": np.nan,
                                            "entropy_thr": np.nan, "recovery_min_conf": rmin,
                                            "rescore_threshold": rt, "rescore_alpha": alpha})
    for c in configs:
        c["method_label"] = method_label(c)
    return configs


# -----------------------------------------------------------------------------
# Evaluation and summaries
# -----------------------------------------------------------------------------

def pair_wbf_seconds(timing_df: pd.DataFrame) -> Dict[str, float]:
    return timing_df.groupby("pair")["total_sec_model1_model2_wbf"].sum().to_dict()


def mean_verifier_seconds(scored_df: pd.DataFrame) -> Dict[str, float]:
    valid = scored_df["crop_valid"].astype(bool)
    clip_sec = float(scored_df.loc[valid, "clip_inference_sec_est"].mean()) if valid.any() else 0.0
    yolo_sec = float(scored_df.loc[valid, "yolo_inference_sec_est"].mean()) if valid.any() else 0.0
    return {"clip": clip_sec, "yolo": yolo_sec, "none": 0.0}


def evaluate_config(scored_df: pd.DataFrame, gt_df: pd.DataFrame, timing_df: pd.DataFrame,
                    cfg: Dict, split: str, args, compute_ap: bool = False) -> Dict:
    pair = cfg["pair"]
    pair_df = scored_df[scored_df["pair"] == pair].copy()
    if cfg["method"].startswith("semantic_rescore"):
        rmin = float(cfg.get("recovery_min_conf", args.rescore_original_min_conf))
        pair_df = pair_df[pair_df["score"] >= rmin].copy()
    keep, verified, recovered, score_override = make_decision(pair_df, cfg, args.duplicate_iou)
    det = pair_df[keep].copy()
    if score_override is not None and score_override in det.columns:
        det["original_hn_wbf_score"] = det["score"]
        det["score"] = det[score_override]
    metrics = greedy_fixed_metrics(det, gt_df, match_iou=args.match_iou)
    base_sec = float(pair_wbf_seconds(timing_df).get(pair, 0.0))
    v_sec_map = mean_verifier_seconds(scored_df)
    verifier = cfg.get("verifier", "none")
    n_verified = int(verified.sum())
    verifier_sec = n_verified * v_sec_map.get(verifier, 0.0)
    total_sec = base_sec + verifier_sec
    n_images = int(gt_df["image_id"].nunique())
    out = {
        "split": split,
        "model_set": "HN",
        "pair": pair,
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
        "hn_wbf_sec": base_sec,
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
    for cfg in tqdm(configs, desc=f"{split}: Phase-10 config grid"):
        rows.append(evaluate_config(scored_df, gt_df, timing_df, cfg, split, args, compute_ap=False))
    return pd.DataFrame(rows)


def select_best_by_pair_method(grid_df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for (pair, method_label), g in grid_df.groupby(["pair", "method_label"], sort=False):
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
        gg = g.sort_values(by=sort_cols, ascending=asc)
        rows.append(gg.iloc[0].to_dict())
    return pd.DataFrame(rows)


def add_map_to_rows(row_df: pd.DataFrame, scored_df: pd.DataFrame, gt_df: pd.DataFrame,
                    timing_df: pd.DataFrame, split: str, args, desc: str) -> pd.DataFrame:
    rows = []
    for _, r in tqdm(row_df.iterrows(), total=len(row_df), desc=desc):
        cfg = {k: r[k] for k in [
            "pair", "method", "method_label", "verifier", "base_conf", "direct_conf", "verifier_prob_thr",
            "entropy_thr", "recovery_min_conf", "rescore_threshold", "rescore_alpha"
        ] if k in r.index}
        rows.append(evaluate_config(scored_df, gt_df, timing_df, cfg, split, args, compute_ap=True))
    return pd.DataFrame(rows)


def apply_valid_selected_to_test(valid_best: pd.DataFrame, test_scored: pd.DataFrame,
                                 test_gt: pd.DataFrame, test_timing: pd.DataFrame, args) -> pd.DataFrame:
    return add_map_to_rows(valid_best, test_scored, test_gt, test_timing, "test", args,
                           "test: frozen Phase-10 configs")


def mean_std_table(df: pd.DataFrame, group_col: str = "method_label") -> pd.DataFrame:
    cols = ["precision", "recall", "f1", "custom_fp_ratio", "fdr", "tp", "fp", "fn", "mAP50", "mAP50_95",
            "end_to_end_images_per_sec_est", "num_verified_candidates", "num_recovered_candidates", "num_kept_detections"]
    rows = []
    for key, g in df.groupby(group_col, sort=False):
        row = {group_col: key, "num_pairs": int(g["pair"].nunique())}
        for c in cols:
            if c in g.columns:
                row[f"{c}_mean"] = float(g[c].mean())
                row[f"{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_wbf_curve(grid_df: pd.DataFrame) -> pd.DataFrame:
    return grid_df[grid_df["method_label"] == "HN_WBF_threshold_curve"].copy()


def matched_recall_table(method_df: pd.DataFrame, wbf_curve: pd.DataFrame, tolerance: float, split: str) -> pd.DataFrame:
    rows = []
    for _, r in method_df.iterrows():
        if r["method_label"] == "HN_WBF_threshold_curve":
            continue
        g = wbf_curve[wbf_curve["pair"] == r["pair"]].copy()
        if len(g) == 0:
            continue
        candidates = g[g["recall"] >= (float(r["recall"]) - tolerance)].copy()
        matched_type = "recall_at_or_above"
        if len(candidates) == 0:
            candidates = g.copy()
            candidates["recall_distance"] = (candidates["recall"] - float(r["recall"])).abs()
            candidates = candidates.sort_values(["recall_distance", "fp", "f1"], ascending=[True, True, False])
            matched_type = "closest_recall_no_hn_wbf_at_or_above"
        else:
            candidates = candidates.sort_values(["fp", "recall", "f1"], ascending=[True, False, False])
        wb = candidates.iloc[0]
        rows.append({
            "split": split, "pair": r["pair"], "method_label": r["method_label"],
            "method_precision": r["precision"], "method_recall": r["recall"], "method_f1": r["f1"],
            "method_fp": int(r["fp"]), "method_fn": int(r["fn"]),
            "hn_wbf_matched_conf": wb["base_conf"], "hn_wbf_precision": wb["precision"],
            "hn_wbf_recall": wb["recall"], "hn_wbf_f1": wb["f1"],
            "hn_wbf_fp": int(wb["fp"]), "hn_wbf_fn": int(wb["fn"]),
            "fp_change_vs_hn_wbf": int(r["fp"]) - int(wb["fp"]),
            "fn_change_vs_hn_wbf": int(r["fn"]) - int(wb["fn"]),
            "f1_change_vs_hn_wbf": float(r["f1"]) - float(wb["f1"]),
            "recall_difference": float(r["recall"]) - float(wb["recall"]),
            "matched_type": matched_type,
        })
    return pd.DataFrame(rows)


def matched_precision_table(method_df: pd.DataFrame, wbf_curve: pd.DataFrame, tolerance: float, split: str) -> pd.DataFrame:
    rows = []
    for _, r in method_df.iterrows():
        if r["method_label"] == "HN_WBF_threshold_curve":
            continue
        g = wbf_curve[wbf_curve["pair"] == r["pair"]].copy()
        if len(g) == 0:
            continue
        candidates = g[g["precision"] >= (float(r["precision"]) - tolerance)].copy()
        matched_type = "precision_at_or_above"
        if len(candidates) == 0:
            candidates = g.copy()
            candidates["precision_distance"] = (candidates["precision"] - float(r["precision"])).abs()
            candidates = candidates.sort_values(["precision_distance", "fn", "f1"], ascending=[True, True, False])
            matched_type = "closest_precision_no_hn_wbf_at_or_above"
        else:
            candidates = candidates.sort_values(["fn", "precision", "f1"], ascending=[True, False, False])
        wb = candidates.iloc[0]
        rows.append({
            "split": split, "pair": r["pair"], "method_label": r["method_label"],
            "method_precision": r["precision"], "method_recall": r["recall"], "method_f1": r["f1"],
            "method_fp": int(r["fp"]), "method_fn": int(r["fn"]),
            "hn_wbf_matched_conf": wb["base_conf"], "hn_wbf_precision": wb["precision"],
            "hn_wbf_recall": wb["recall"], "hn_wbf_f1": wb["f1"],
            "hn_wbf_fp": int(wb["fp"]), "hn_wbf_fn": int(wb["fn"]),
            "fp_change_vs_hn_wbf": int(r["fp"]) - int(wb["fp"]),
            "fn_change_vs_hn_wbf": int(r["fn"]) - int(wb["fn"]),
            "f1_change_vs_hn_wbf": float(r["f1"]) - float(wb["f1"]),
            "precision_difference": float(r["precision"]) - float(wb["precision"]),
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


def load_existing_or_score(split: str, phase9_dir: Path, out_dir: Path, args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores_path = out_dir / f"phase10_{split}_HN_candidate_verifier_scores.csv"
    gt_path = phase9_dir / f"phase9_{split}_ground_truth.csv"
    timing_path = phase9_dir / f"phase9_{split}_HN_timing.csv"
    if args.reuse_scores and scores_path.exists():
        print(f"Reusing scored {split} candidates: {scores_path}")
        return load_scored_split(scores_path, gt_path, timing_path, split)
    cand_path = phase9_dir / f"phase9_{split}_HN_wbf_candidates.csv"
    if not cand_path.exists():
        raise FileNotFoundError(f"Missing Phase-9 HN-WBF candidates: {cand_path}")
    cand = standardize_hn_candidates(pd.read_csv(cand_path))
    if args.limit and args.limit > 0:
        image_ids = cand["image_id"].drop_duplicates().head(args.limit).tolist()
        cand = cand[cand["image_id"].isin(image_ids)].reset_index(drop=True)
    scored, _ = run_verifier_scoring(cand, args, out_dir, split)
    return load_scored_split(scores_path, gt_path, timing_path, split)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 10 HN-WBF + CLIP/YOLO verifier final analysis")
    parser.add_argument("--phase9_dir", type=str, default="/home/inspur/ishrai/phase9_HN_detector_wbf")
    parser.add_argument("--out_dir", type=str, default="/home/inspur/ishrai/phase10_HN_wbf_verifier")
    parser.add_argument("--clip_ckpt", type=str, default="/home/inspur/ishrai/all_togather/clip_verifier/best_clip_finetuned.pth")
    parser.add_argument("--clip_pretrained", type=str, default="/home/inspur/ishrai/pretrained_openclip/ViT-B-32_laion2b_e16/open_clip_model.safetensors")
    parser.add_argument("--clip_model_name", type=str, default="ViT-B-32")
    parser.add_argument("--yolo_cls_ckpt", type=str, default="/home/inspur/ishrai/all_togather/yolo_verifier/YOLOv8l_cls_best.pt")
    parser.add_argument("--yolo_cls_imgsz", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--crop_pad", type=float, default=0.05)
    parser.add_argument("--min_crop_size", type=int, default=4)
    parser.add_argument("--reuse_scores", action="store_true", help="Reuse phase10_*_candidate_verifier_scores.csv if they already exist.")
    parser.add_argument("--limit", type=int, default=0, help="Optional debug limit in number of images per split. Use 0 for full run.")

    parser.add_argument("--fixed_base_conf", type=float, default=0.25)
    parser.add_argument("--wbf_threshold_grid", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--direct_conf_grid", type=str, default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--verifier_prob_grid", type=str, default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--entropy_grid", type=str, default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--recovery_min_conf_grid", type=str, default="0.05,0.10,0.15,0.20")
    parser.add_argument("--rescore_threshold_grid", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--rescore_alpha_grid", type=str, default="0.50,0.70")
    parser.add_argument("--rescore_original_min_conf", type=float, default=0.05)
    parser.add_argument("--skip_rescore", action="store_true", help="Skip semantic rescoring modes for a faster run.")
    parser.add_argument("--duplicate_iou", type=float, default=0.50)
    parser.add_argument("--match_iou", type=float, default=0.50)
    parser.add_argument("--matched_tolerance", type=float, default=0.002)
    parser.add_argument("--selected_pair", type=str, default="y8x+y8l")
    parser.add_argument("--limit_pairs", type=str, default="", help="Optional comma-separated pair list for debug run.")
    parser.add_argument("--selection_mode", type=str, default="balanced", choices=["balanced", "fp_control", "fn_recovery"])
    parser.add_argument("--final_candidate_mode", type=str, default="balanced", choices=["balanced", "fp_control", "fn_recovery"])
    parser.add_argument("--top_k_final_candidates", type=int, default=20)

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    phase9_dir = Path(args.phase9_dir)
    ensure_dir(out_dir)

    print("Loading or scoring validation HN-WBF candidates...")
    valid_scored, valid_gt, valid_timing = load_existing_or_score("valid", phase9_dir, out_dir, args)
    print("Loading or scoring test HN-WBF candidates...")
    test_scored, test_gt, test_timing = load_existing_or_score("test", phase9_dir, out_dir, args)

    pairs = sorted(valid_scored["pair"].dropna().unique().tolist())
    test_pairs = set(test_scored["pair"].dropna().unique().tolist())
    pairs = [p for p in pairs if p in test_pairs]
    if args.limit_pairs.strip():
        wanted = [x.strip() for x in args.limit_pairs.split(',') if x.strip()]
        pairs = [p for p in pairs if p in wanted]
        valid_scored = valid_scored[valid_scored["pair"].isin(pairs)].reset_index(drop=True)
        test_scored = test_scored[test_scored["pair"].isin(pairs)].reset_index(drop=True)
        valid_timing = valid_timing[valid_timing["pair"].isin(pairs)].reset_index(drop=True)
        test_timing = test_timing[test_timing["pair"].isin(pairs)].reset_index(drop=True)
    if not pairs:
        raise RuntimeError("No common HN-WBF pairs found between validation and test.")

    configs = build_configs(pairs, args)
    print(f"HN-WBF pairs: {pairs}")
    print(f"Total Phase-10 validation configs: {len(configs)}")
    print("Validation selects thresholds. Test applies frozen validation-selected settings.")

    settings = vars(args).copy()
    settings.update({
        "phase": "Phase 10 HN-WBF + verifier final analysis",
        "pairs": pairs,
        "valid_num_candidates": int(len(valid_scored)),
        "test_num_candidates": int(len(test_scored)),
        "valid_num_images": int(valid_gt["image_id"].nunique()),
        "test_num_images": int(test_gt["image_id"].nunique()),
        "valid_num_gt": int(len(valid_gt)),
        "test_num_gt": int(len(test_gt)),
        "detection_classes": WEAPON_CLASSES,
        "verifier_classes": UNIFIED_CLASSES,
        "test_policy": "Validation selects configurations; test is frozen evaluation only.",
    })
    safe_json_dump(settings, out_dir / "phase10_HN_wbf_verifier_settings.json")

    # Validation full grid.
    valid_grid = evaluate_grid(valid_scored, valid_gt, valid_timing, configs, "valid", args)
    valid_grid.to_csv(out_dir / "phase10_valid_full_grid.csv", index=False)
    valid_grid.to_excel(out_dir / "phase10_valid_full_grid.xlsx", index=False)

    # Best validation config per pair/method + mAP for those rows only.
    valid_best_raw = select_best_by_pair_method(valid_grid, args)
    valid_best = add_map_to_rows(valid_best_raw, valid_scored, valid_gt, valid_timing, "valid", args,
                                  "valid: mAP for selected Phase-10 pair/method configs")
    valid_best.to_csv(out_dir / "phase10_valid_best_by_pair_method.csv", index=False)
    valid_best.to_excel(out_dir / "phase10_valid_best_by_pair_method.xlsx", index=False)

    # Apply validation-selected configs to test.
    test_frozen = apply_valid_selected_to_test(valid_best, test_scored, test_gt, test_timing, args)
    test_frozen.to_csv(out_dir / "phase10_test_frozen_results.csv", index=False)
    test_frozen.to_excel(out_dir / "phase10_test_frozen_results.xlsx", index=False)

    # HN-WBF threshold curve on test for matched operating point analysis only.
    wbf_test_configs = [c for c in configs if c["method_label"] == "HN_WBF_threshold_curve"]
    test_wbf_curve = evaluate_grid(test_scored, test_gt, test_timing, wbf_test_configs, "test", args)
    test_wbf_curve.to_csv(out_dir / "phase10_test_HN_wbf_threshold_curve.csv", index=False)
    test_wbf_curve.to_excel(out_dir / "phase10_test_HN_wbf_threshold_curve.xlsx", index=False)

    valid_meanstd = mean_std_table(valid_best)
    test_meanstd = mean_std_table(test_frozen)
    valid_meanstd.to_csv(out_dir / "phase10_valid_mean_std_across_pairs.csv", index=False)
    test_meanstd.to_csv(out_dir / "phase10_test_mean_std_across_pairs.csv", index=False)
    valid_meanstd.to_excel(out_dir / "phase10_valid_mean_std_across_pairs.xlsx", index=False)
    test_meanstd.to_excel(out_dir / "phase10_test_mean_std_across_pairs.xlsx", index=False)

    valid_wbf_curve = build_wbf_curve(valid_grid)
    valid_wbf_curve.to_csv(out_dir / "phase10_valid_HN_wbf_threshold_curve.csv", index=False)
    valid_wbf_curve.to_excel(out_dir / "phase10_valid_HN_wbf_threshold_curve.xlsx", index=False)

    valid_matched_recall = matched_recall_table(valid_best, valid_wbf_curve, args.matched_tolerance, "valid")
    valid_matched_precision = matched_precision_table(valid_best, valid_wbf_curve, args.matched_tolerance, "valid")
    test_matched_recall = matched_recall_table(test_frozen, test_wbf_curve, args.matched_tolerance, "test")
    test_matched_precision = matched_precision_table(test_frozen, test_wbf_curve, args.matched_tolerance, "test")

    valid_matched_recall.to_csv(out_dir / "phase10_valid_matched_recall_vs_HN_wbf.csv", index=False)
    valid_matched_precision.to_csv(out_dir / "phase10_valid_matched_precision_vs_HN_wbf.csv", index=False)
    test_matched_recall.to_csv(out_dir / "phase10_test_matched_recall_vs_HN_wbf.csv", index=False)
    test_matched_precision.to_csv(out_dir / "phase10_test_matched_precision_vs_HN_wbf.csv", index=False)
    valid_matched_recall.to_excel(out_dir / "phase10_valid_matched_recall_vs_HN_wbf.xlsx", index=False)
    valid_matched_precision.to_excel(out_dir / "phase10_valid_matched_precision_vs_HN_wbf.xlsx", index=False)
    test_matched_recall.to_excel(out_dir / "phase10_test_matched_recall_vs_HN_wbf.xlsx", index=False)
    test_matched_precision.to_excel(out_dir / "phase10_test_matched_precision_vs_HN_wbf.xlsx", index=False)

    top_valid_candidates = candidate_final_configs(valid_best, args)
    key_cols = ["pair", "method_label", "method", "verifier", "base_conf", "direct_conf", "verifier_prob_thr",
                "entropy_thr", "recovery_min_conf", "rescore_threshold", "rescore_alpha"]
    if len(top_valid_candidates):
        top_keys = top_valid_candidates[key_cols].copy()
        top_test_results = test_frozen.merge(top_keys, on=key_cols, how="inner")
        top_test_results = top_test_results.sort_values(["f1", "fp", "recall"], ascending=[False, True, False])
    else:
        top_test_results = pd.DataFrame()

    selected_pair_valid = valid_best[valid_best["pair"] == args.selected_pair].sort_values(
        ["f1", "fp", "recall"], ascending=[False, True, False]
    )
    selected_pair_test = test_frozen[test_frozen["pair"] == args.selected_pair].sort_values(
        ["f1", "fp", "recall"], ascending=[False, True, False]
    )

    top_valid_candidates.to_csv(out_dir / "phase10_top_validation_final_candidates.csv", index=False)
    top_test_results.to_csv(out_dir / "phase10_top_validation_candidates_on_test.csv", index=False)
    selected_pair_valid.to_csv(out_dir / "phase10_selected_pair_valid_results.csv", index=False)
    selected_pair_test.to_csv(out_dir / "phase10_selected_pair_test_results.csv", index=False)
    top_valid_candidates.to_excel(out_dir / "phase10_top_validation_final_candidates.xlsx", index=False)
    top_test_results.to_excel(out_dir / "phase10_top_validation_candidates_on_test.xlsx", index=False)
    selected_pair_valid.to_excel(out_dir / "phase10_selected_pair_valid_results.xlsx", index=False)
    selected_pair_test.to_excel(out_dir / "phase10_selected_pair_test_results.xlsx", index=False)

    paper_xlsx = out_dir / "phase10_HN_wbf_verifier_paper_summary.xlsx"
    with pd.ExcelWriter(paper_xlsx) as writer:
        top_valid_candidates.to_excel(writer, sheet_name="top_valid_candidates", index=False)
        top_test_results.to_excel(writer, sheet_name="top_candidates_on_test", index=False)
        selected_pair_valid.to_excel(writer, sheet_name="selected_pair_valid", index=False)
        selected_pair_test.to_excel(writer, sheet_name="selected_pair_test", index=False)
        valid_best.sort_values(["pair", "method_label"]).to_excel(writer, sheet_name="valid_best_by_method", index=False)
        test_frozen.sort_values(["pair", "method_label"]).to_excel(writer, sheet_name="test_frozen_by_method", index=False)
        valid_meanstd.to_excel(writer, sheet_name="valid_mean_std", index=False)
        test_meanstd.to_excel(writer, sheet_name="test_mean_std", index=False)
        valid_matched_recall.to_excel(writer, sheet_name="valid_matched_recall", index=False)
        test_matched_recall.to_excel(writer, sheet_name="test_matched_recall", index=False)
        valid_matched_precision.to_excel(writer, sheet_name="valid_matched_precision", index=False)
        test_matched_precision.to_excel(writer, sheet_name="test_matched_precision", index=False)
        valid_wbf_curve.to_excel(writer, sheet_name="valid_HN_wbf_curve", index=False)
        test_wbf_curve.to_excel(writer, sheet_name="test_HN_wbf_curve", index=False)

    print("\n" + "=" * 80)
    print("PHASE 10 COMPLETE")
    print(f"Output folder                         : {out_dir}")
    print(f"Paper summary                         : {paper_xlsx}")
    print(f"Validation full grid                  : {out_dir / 'phase10_valid_full_grid.xlsx'}")
    print(f"Validation best by pair/method         : {out_dir / 'phase10_valid_best_by_pair_method.xlsx'}")
    print(f"Test frozen results                    : {out_dir / 'phase10_test_frozen_results.xlsx'}")
    print(f"Mean/std across pairs                  : {out_dir / 'phase10_test_mean_std_across_pairs.xlsx'}")
    print("=" * 80)
    print("\nTop validation candidates:")
    cols = ["pair", "method_label", "precision", "recall", "f1", "fp", "fn", "base_conf", "direct_conf", "verifier_prob_thr", "entropy_thr", "recovery_min_conf", "end_to_end_images_per_sec_est"]
    show_cols = [c for c in cols if c in top_valid_candidates.columns]
    print(top_valid_candidates[show_cols].head(20).to_string(index=False) if len(top_valid_candidates) else "No verifier candidates found.")


if __name__ == "__main__":
    main()
