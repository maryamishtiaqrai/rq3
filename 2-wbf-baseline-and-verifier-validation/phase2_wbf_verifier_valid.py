#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: WBF + verifier validation experiments.

This script DOES NOT rerun YOLO detection or WBF.
It uses the saved Phase-1 WBF fused candidates, crops each candidate box,
runs the trained CLIP verifier and the trained YOLO classification verifier,
and evaluates verifier-based decision rules on the validation split.

Outputs:
  1) Candidate-level verifier probabilities/entropy
  2) Full threshold grid results
  3) Best validation configuration per detector pair and method
  4) Pair-level timing / estimated throughput
  5) Settings JSON for reproducibility

Expected Phase-1 input files:
  phase1_wbf_only_valid_all_fused_candidates.csv
  phase1_wbf_only_valid_ground_truth.csv
  phase1_wbf_only_valid_timing_by_image.csv
  phase1_wbf_only_valid_settings.json
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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


UNIFIED_CLASSES = ["knife", "pistol", "non_weapon"]
WEAPON_CLASSES = ["knife", "pistol"]
EPS = 1e-12


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_json_dump(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_float_grid(text: str) -> List[float]:
    vals = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals


def canonical_name(x) -> str:
    s = str(x).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    if s in ["background", "nonweapon", "non_weapon_object", "non_weapon_objects", "negative", "bg"]:
        return "non_weapon"
    return s


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
    """VOC/COCO-style precision envelope AP."""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def maybe_abs_to_norm_box(row, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Accept normalized [0,1] boxes or absolute pixel boxes and return normalized."""
    x1, y1, x2, y2 = map(float, [row["x1"], row["y1"], row["x2"], row["y2"]])
    # Existing Phase-1 boxes are normalized. This fallback protects against future runs.
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
    for cname, pk, pp, pn in zip(df["class_name"], df[f"{prefix}_prob_knife"],
                                df[f"{prefix}_prob_pistol"], df[f"{prefix}_prob_non_weapon"]):
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


# -----------------------------------------------------------------------------
# CLIP verifier
# -----------------------------------------------------------------------------

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
        print("WARNING: Many missing/unexpected keys. Check that the checkpoint matches ViT-B-32 CLIP.")
        print("First missing keys:", list(missing)[:10])
        print("First unexpected keys:", list(unexpected)[:10])

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


# -----------------------------------------------------------------------------
# YOLO classification verifier
# -----------------------------------------------------------------------------

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
        raise RuntimeError(
            f"YOLO-cls model.names cannot be mapped to {UNIFIED_CLASSES}. Missing: {missing}. names={names}"
        )
    return model, canon_to_idx


def score_yolo_cls_batch(model, canon_to_idx: Dict[str, int], crops: List[Image.Image], args) -> np.ndarray:
    # Ultralytics accepts numpy RGB images in a list.
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
            raise RuntimeError("YOLO classification result has no r.probs. Check that the checkpoint is a classification model.")
        p = r.probs.data.detach().cpu().numpy().astype(np.float32)
        aligned = np.zeros(3, dtype=np.float32)
        aligned[0] = p[canon_to_idx["knife"]]
        aligned[1] = p[canon_to_idx["pistol"]]
        aligned[2] = p[canon_to_idx["non_weapon"]]
        out.append(aligned)
    return np.vstack(out).astype(np.float32)


# -----------------------------------------------------------------------------
# Candidate scoring
# -----------------------------------------------------------------------------

def run_verifier_scoring(cand_df: pd.DataFrame, args, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score all saved WBF candidates with CLIP and YOLO-cls verifiers."""
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    clip_model, clip_preprocess, clip_text_features = load_clip_verifier(args, device)
    yolo_cls_model, yolo_cls_map = load_yolo_cls(args)

    scored_rows = []
    timing_rows = []

    total = len(cand_df)
    print(f"Scoring {total} WBF fused candidates.")

    # Process by image to read each image only once.
    grouped = cand_df.groupby("image_path", sort=False)
    batch_crops, batch_indices = [], []
    batch_meta = []

    def flush_batch():
        nonlocal batch_crops, batch_indices, batch_meta, scored_rows, timing_rows
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
            "batch_size": n,
            "clip_batch_sec": clip_sec,
            "clip_sec_per_crop": clip_per_crop,
            "clip_crops_per_sec": n / clip_sec if clip_sec > 0 else np.nan,
            "yolo_batch_sec": yolo_sec,
            "yolo_sec_per_crop": yolo_per_crop,
            "yolo_crops_per_sec": n / yolo_sec if yolo_sec > 0 else np.nan,
        })

        batch_crops, batch_indices, batch_meta = [], [], []

    for image_path, g in tqdm(grouped, total=cand_df["image_path"].nunique(), desc="Images"):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"WARNING: cannot read image: {image_path}")
            for _, row in g.iterrows():
                base = row.to_dict()
                base.update({"crop_valid": False})
                scored_rows.append(base)
            continue
        h, w = image_bgr.shape[:2]
        for idx, row in g.iterrows():
            box = maybe_abs_to_norm_box(row, w, h)
            crop = crop_candidate(image_bgr, box, pad=args.crop_pad, min_size=args.min_crop_size)
            if crop is None:
                base = row.to_dict()
                base.update({"crop_valid": False})
                scored_rows.append(base)
                continue
            batch_crops.append(crop)
            batch_indices.append(idx)
            batch_meta.append(row.to_dict())
            if len(batch_crops) >= args.batch_size:
                flush_batch()

    flush_batch()

    scored = pd.DataFrame(scored_rows)
    scored = add_detector_class_probability(scored, "clip")
    scored = add_detector_class_probability(scored, "yolo")
    timing = pd.DataFrame(timing_rows)

    # Save immediately because this is the expensive output.
    scored_csv = out_dir / "phase2_valid_candidate_verifier_scores.csv"
    scored_xlsx = out_dir / "phase2_valid_candidate_verifier_scores.xlsx"
    timing_csv = out_dir / "phase2_valid_verifier_batch_timing.csv"
    timing_xlsx = out_dir / "phase2_valid_verifier_batch_timing.xlsx"

    scored.to_csv(scored_csv, index=False)
    scored.to_excel(scored_xlsx, index=False)
    timing.to_csv(timing_csv, index=False)
    timing.to_excel(timing_xlsx, index=False)

    print(f"Saved candidate verifier scores: {scored_csv}")
    print(f"Saved candidate verifier scores Excel: {scored_xlsx}")
    return scored, timing


# -----------------------------------------------------------------------------
# Detection evaluation
# -----------------------------------------------------------------------------

def build_gt_index(gt_df: pd.DataFrame):
    gt_index = {}
    for i, r in gt_df.reset_index(drop=True).iterrows():
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
    fp_ratio = fp / (tp + fp) if (tp + fp) else 0.0

    out = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "custom_fp_ratio": fp_ratio,
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
        cls_aps = []
        for t in thresholds:
            cls_aps.append(ap_for_class(det_df, gt_df, c, float(t)))
        out[f"ap50_95_{c}"] = float(np.nanmean(cls_aps))
        all_aps.extend(cls_aps)
    out["mAP50"] = float(np.nanmean(ap50s))
    out["mAP50_95"] = float(np.nanmean(all_aps))
    return out


def choose_best(grid_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pair, method), g in grid_df.groupby(["pair", "method"], sort=False):
        # Main selection: highest F1. Tie breakers: lower FP, higher recall, lower estimated time.
        gg = g.sort_values(
            by=["f1", "fp", "recall", "end_to_end_images_per_sec_est"],
            ascending=[False, True, False, False],
        )
        rows.append(gg.iloc[0].to_dict())
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Method grids
# -----------------------------------------------------------------------------

def make_keep_mask(df: pd.DataFrame, method: str, verifier: Optional[str],
                   base_conf: float, direct_conf: Optional[float] = None,
                   ver_prob: Optional[float] = None, ent_thr: Optional[float] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Return accepted mask and verified mask.
    verified mask means candidates that would need verifier inference in real operation.
    """
    valid = df.get("crop_valid", True)
    if isinstance(valid, bool):
        valid = pd.Series([valid] * len(df), index=df.index)
    base = (df["score"] >= base_conf) & valid

    if method == "WBF_only":
        keep = df["score"] >= base_conf
        verified = pd.Series([False] * len(df), index=df.index)
        return keep, verified

    if verifier not in ["clip", "yolo"]:
        raise ValueError("verifier must be 'clip' or 'yolo' for verifier methods")

    same = df[f"{verifier}_same_as_detector"].astype(bool)
    prob = df[f"{verifier}_prob_detector_class"].astype(float)
    ent = df[f"{verifier}_entropy"].astype(float)

    if method.endswith("hard"):
        verified = base
        keep = base & same
        return keep, verified

    if method.endswith("selective_conf"):
        if direct_conf is None or ver_prob is None:
            raise ValueError("selective_conf requires direct_conf and ver_prob")
        direct = base & (df["score"] >= direct_conf)
        need_ver = base & (df["score"] < direct_conf)
        keep = direct | (need_ver & same & (prob >= ver_prob))
        return keep, need_ver

    if method.endswith("selective_entropy"):
        if direct_conf is None or ent_thr is None:
            raise ValueError("selective_entropy requires direct_conf and ent_thr")
        direct = base & (df["score"] >= direct_conf)
        need_ver = base & (df["score"] < direct_conf)
        keep = direct | (need_ver & same & (ent <= ent_thr))
        return keep, need_ver

    raise ValueError(f"Unknown method: {method}")


def evaluate_phase2(scored_df: pd.DataFrame, gt_df: pd.DataFrame, phase1_timing: pd.DataFrame,
                    args, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_grid = parse_float_grid(args.base_conf_grid)
    direct_grid = parse_float_grid(args.direct_conf_grid)
    ver_prob_grid = parse_float_grid(args.ver_prob_grid)
    entropy_grid = parse_float_grid(args.entropy_grid)

    n_images = int(gt_df["image_id"].nunique())
    # Phase-1 WBF end-to-end timing by pair.
    p1_time = phase1_timing.groupby("pair")["total_sec_model1_model2_wbf"].sum().to_dict()

    mean_clip_sec_per_crop = float(scored_df.loc[scored_df["crop_valid"] == True, "clip_inference_sec_est"].mean())
    mean_yolo_sec_per_crop = float(scored_df.loc[scored_df["crop_valid"] == True, "yolo_inference_sec_est"].mean())
    verifier_sec = {"clip": mean_clip_sec_per_crop, "yolo": mean_yolo_sec_per_crop, None: 0.0}

    print(f"Mean CLIP sec/crop: {mean_clip_sec_per_crop:.6f}")
    print(f"Mean YOLO-cls sec/crop: {mean_yolo_sec_per_crop:.6f}")

    grid_rows = []

    pairs = sorted(scored_df["pair"].dropna().unique().tolist())
    for pair in tqdm(pairs, desc="Evaluating pairs"):
        pair_df = scored_df[scored_df["pair"] == pair].copy()
        p1_sec = float(p1_time.get(pair, np.nan))

        # WBF only confidence sweep.
        for base_conf in base_grid:
            keep, verified = make_keep_mask(pair_df, "WBF_only", None, base_conf)
            det = pair_df[keep].copy()
            metrics = greedy_fixed_metrics(det, gt_df, match_iou=args.match_iou)
            total_sec_est = p1_sec
            grid_rows.append({
                "pair": pair,
                "method": "WBF_only",
                "verifier": "none",
                "base_conf": base_conf,
                "direct_conf": np.nan,
                "verifier_prob_thr": np.nan,
                "entropy_thr": np.nan,
                "num_verified_candidates": int(verified.sum()),
                "phase1_wbf_sec": p1_sec,
                "verifier_sec_est": 0.0,
                "end_to_end_sec_est": total_sec_est,
                "end_to_end_images_per_sec_est": n_images / total_sec_est if total_sec_est > 0 else np.nan,
                **metrics,
            })

        for verifier in ["clip", "yolo"]:
            # Hard verifier: verify every candidate above base threshold.
            method = f"{verifier.upper()}_hard"
            for base_conf in base_grid:
                keep, verified = make_keep_mask(pair_df, method, verifier, base_conf)
                det = pair_df[keep].copy()
                metrics = greedy_fixed_metrics(det, gt_df, match_iou=args.match_iou)
                nver = int(verified.sum())
                vsec = nver * verifier_sec[verifier]
                total_sec_est = p1_sec + vsec
                grid_rows.append({
                    "pair": pair,
                    "method": method,
                    "verifier": verifier,
                    "base_conf": base_conf,
                    "direct_conf": np.nan,
                    "verifier_prob_thr": np.nan,
                    "entropy_thr": np.nan,
                    "num_verified_candidates": nver,
                    "phase1_wbf_sec": p1_sec,
                    "verifier_sec_est": vsec,
                    "end_to_end_sec_est": total_sec_est,
                    "end_to_end_images_per_sec_est": n_images / total_sec_est if total_sec_est > 0 else np.nan,
                    **metrics,
                })

            # Selective confidence verifier.
            method = f"{verifier.upper()}_selective_conf"
            for base_conf in base_grid:
                for direct_conf in direct_grid:
                    if direct_conf < base_conf:
                        continue
                    for ver_prob in ver_prob_grid:
                        keep, verified = make_keep_mask(
                            pair_df, method, verifier,
                            base_conf=base_conf,
                            direct_conf=direct_conf,
                            ver_prob=ver_prob,
                        )
                        det = pair_df[keep].copy()
                        metrics = greedy_fixed_metrics(det, gt_df, match_iou=args.match_iou)
                        nver = int(verified.sum())
                        vsec = nver * verifier_sec[verifier]
                        total_sec_est = p1_sec + vsec
                        grid_rows.append({
                            "pair": pair,
                            "method": method,
                            "verifier": verifier,
                            "base_conf": base_conf,
                            "direct_conf": direct_conf,
                            "verifier_prob_thr": ver_prob,
                            "entropy_thr": np.nan,
                            "num_verified_candidates": nver,
                            "phase1_wbf_sec": p1_sec,
                            "verifier_sec_est": vsec,
                            "end_to_end_sec_est": total_sec_est,
                            "end_to_end_images_per_sec_est": n_images / total_sec_est if total_sec_est > 0 else np.nan,
                            **metrics,
                        })

            # Selective entropy verifier.
            method = f"{verifier.upper()}_selective_entropy"
            for base_conf in base_grid:
                for direct_conf in direct_grid:
                    if direct_conf < base_conf:
                        continue
                    for ent_thr in entropy_grid:
                        keep, verified = make_keep_mask(
                            pair_df, method, verifier,
                            base_conf=base_conf,
                            direct_conf=direct_conf,
                            ent_thr=ent_thr,
                        )
                        det = pair_df[keep].copy()
                        metrics = greedy_fixed_metrics(det, gt_df, match_iou=args.match_iou)
                        nver = int(verified.sum())
                        vsec = nver * verifier_sec[verifier]
                        total_sec_est = p1_sec + vsec
                        grid_rows.append({
                            "pair": pair,
                            "method": method,
                            "verifier": verifier,
                            "base_conf": base_conf,
                            "direct_conf": direct_conf,
                            "verifier_prob_thr": np.nan,
                            "entropy_thr": ent_thr,
                            "num_verified_candidates": nver,
                            "phase1_wbf_sec": p1_sec,
                            "verifier_sec_est": vsec,
                            "end_to_end_sec_est": total_sec_est,
                            "end_to_end_images_per_sec_est": n_images / total_sec_est if total_sec_est > 0 else np.nan,
                            **metrics,
                        })

    grid_df = pd.DataFrame(grid_rows)
    best_df = choose_best(grid_df)

    # Compute AP/mAP only for the selected best configurations to keep runtime reasonable.
    map_rows = []
    for _, row in tqdm(best_df.iterrows(), total=len(best_df), desc="Computing mAP for best configs"):
        pair = row["pair"]
        method = row["method"]
        verifier = row["verifier"] if row["verifier"] != "none" else None
        pair_df = scored_df[scored_df["pair"] == pair].copy()
        keep, verified = make_keep_mask(
            pair_df,
            method=method,
            verifier=verifier,
            base_conf=float(row["base_conf"]),
            direct_conf=None if pd.isna(row["direct_conf"]) else float(row["direct_conf"]),
            ver_prob=None if pd.isna(row["verifier_prob_thr"]) else float(row["verifier_prob_thr"]),
            ent_thr=None if pd.isna(row["entropy_thr"]) else float(row["entropy_thr"]),
        )
        det = pair_df[keep].copy()
        m = compute_map(det, gt_df)
        map_rows.append({"pair": pair, "method": method, **m})

    map_df = pd.DataFrame(map_rows)
    best_df = best_df.merge(map_df, on=["pair", "method"], how="left")

    # Overall mean/std across six pairs for each method using each pair's best validation config.
    agg_cols = [
        "precision", "recall", "f1", "custom_fp_ratio",
        "tp", "fp", "fn", "mAP50", "mAP50_95",
        "end_to_end_images_per_sec_est", "num_verified_candidates",
    ]
    agg_rows = []
    for method, g in best_df.groupby("method", sort=False):
        row = {"method": method, "num_pairs": len(g)}
        for c in agg_cols:
            if c in g.columns:
                row[f"{c}_mean"] = float(g[c].mean())
                row[f"{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)

    grid_csv = out_dir / "phase2_valid_full_threshold_grid.csv"
    grid_xlsx = out_dir / "phase2_valid_full_threshold_grid.xlsx"
    best_csv = out_dir / "phase2_valid_best_by_pair_method.csv"
    best_xlsx = out_dir / "phase2_valid_best_by_pair_method.xlsx"
    agg_csv = out_dir / "phase2_valid_method_mean_std_across_pairs.csv"
    agg_xlsx = out_dir / "phase2_valid_method_mean_std_across_pairs.xlsx"

    grid_df.to_csv(grid_csv, index=False)
    grid_df.to_excel(grid_xlsx, index=False)
    best_df.to_csv(best_csv, index=False)
    best_df.to_excel(best_xlsx, index=False)
    agg_df.to_csv(agg_csv, index=False)
    agg_df.to_excel(agg_xlsx, index=False)

    print(f"Saved full grid: {grid_csv}")
    print(f"Saved best configs: {best_xlsx}")
    print(f"Saved mean/std across pairs: {agg_xlsx}")
    return grid_df, best_df, agg_df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--phase1_dir", type=str, default="/home/inspur/ishrai/phase1_wbf_valid")
    parser.add_argument("--out_dir", type=str, default="/home/inspur/ishrai/phase2_wbf_verifiers_valid")

    parser.add_argument("--clip_ckpt", type=str,
                        default="/home/inspur/ishrai/all_togather/clip_verifier/best_clip_finetuned.pth")
    parser.add_argument("--clip_model_name", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str,
                        default="/home/inspur/ishrai/pretrained_openclip/ViT-B-32_laion2b_e16/open_clip_model.safetensors")

    parser.add_argument("--yolo_cls_ckpt", type=str,
                        default="/home/inspur/ishrai/all_togather/yolo_verifier/YOLOv8l_cls_best.pt")
    parser.add_argument("--yolo_cls_imgsz", type=int, default=224)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--crop_pad", type=float, default=0.05)
    parser.add_argument("--min_crop_size", type=int, default=4)
    parser.add_argument("--match_iou", type=float, default=0.50)

    # Validation grid. You can narrow later after first run.
    parser.add_argument("--base_conf_grid", type=str,
                        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--direct_conf_grid", type=str,
                        default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--ver_prob_grid", type=str,
                        default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--entropy_grid", type=str,
                        default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")

    parser.add_argument("--reuse_scores", action="store_true",
                        help="Use existing phase2_valid_candidate_verifier_scores.csv if it already exists.")

    args = parser.parse_args()

    phase1_dir = Path(args.phase1_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    cand_path = phase1_dir / "phase1_wbf_only_valid_all_fused_candidates.csv"
    gt_path = phase1_dir / "phase1_wbf_only_valid_ground_truth.csv"
    timing_path = phase1_dir / "phase1_wbf_only_valid_timing_by_image.csv"
    p1_settings_path = phase1_dir / "phase1_wbf_only_valid_settings.json"

    for p in [cand_path, gt_path, timing_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required Phase-1 file not found: {p}")

    cand_df = pd.read_csv(cand_path)
    gt_df = pd.read_csv(gt_path)
    phase1_timing = pd.read_csv(timing_path)

    # Normalize class names for safety.
    cand_df["class_name"] = cand_df["class_name"].map(canonical_name)
    gt_df["class_name"] = gt_df["class_name"].map(canonical_name)

    # Use only weapon detection candidates.
    cand_df = cand_df[cand_df["class_name"].isin(WEAPON_CLASSES)].reset_index(drop=True)
    gt_df = gt_df[gt_df["class_name"].isin(WEAPON_CLASSES)].reset_index(drop=True)

    run_settings = vars(args).copy()
    if p1_settings_path.exists():
        with open(p1_settings_path, "r", encoding="utf-8") as f:
            run_settings["phase1_settings"] = json.load(f)
    run_settings["unified_verifier_classes"] = UNIFIED_CLASSES
    run_settings["weapon_detection_classes"] = WEAPON_CLASSES
    run_settings["note"] = (
        "Phase 2 uses validation split only. It does not rerun detection/WBF. "
        "Verifier thresholds and method choices selected here must be frozen before final test evaluation."
    )
    safe_json_dump(run_settings, out_dir / "phase2_valid_settings.json")

    scores_path = out_dir / "phase2_valid_candidate_verifier_scores.csv"
    if args.reuse_scores and scores_path.exists():
        print(f"Reusing existing verifier scores: {scores_path}")
        scored_df = pd.read_csv(scores_path)
    else:
        scored_df, verifier_timing = run_verifier_scoring(cand_df, args, out_dir)

    grid_df, best_df, agg_df = evaluate_phase2(scored_df, gt_df, phase1_timing, args, out_dir)

    print("\n" + "=" * 80)
    print("PHASE 2 VALIDATION COMPLETE")
    print(f"Output folder                         : {out_dir}")
    print(f"Candidate verifier scores CSV         : {out_dir / 'phase2_valid_candidate_verifier_scores.csv'}")
    print(f"Candidate verifier scores Excel       : {out_dir / 'phase2_valid_candidate_verifier_scores.xlsx'}")
    print(f"Full threshold grid Excel             : {out_dir / 'phase2_valid_full_threshold_grid.xlsx'}")
    print(f"Best by pair/method Excel             : {out_dir / 'phase2_valid_best_by_pair_method.xlsx'}")
    print(f"Mean/std across pairs Excel           : {out_dir / 'phase2_valid_method_mean_std_across_pairs.xlsx'}")
    print(f"Settings JSON                         : {out_dir / 'phase2_valid_settings.json'}")
    print("=" * 80)

    print("\nBest validation configurations, first rows:")
    cols = [
        "pair", "method", "precision", "recall", "f1", "fp", "fn", "custom_fp_ratio",
        "base_conf", "direct_conf", "verifier_prob_thr", "entropy_thr",
        "end_to_end_images_per_sec_est", "mAP50", "mAP50_95",
    ]
    cols = [c for c in cols if c in best_df.columns]
    print(best_df[cols].sort_values(["f1", "fp"], ascending=[False, True]).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
