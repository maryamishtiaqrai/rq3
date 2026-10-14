#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5: operating-point diagnostics for WBF + CLIP / YOLO verifier.

Purpose
-------
This phase is designed to answer reviewer-risk questions before writing the paper:

1) Is the verifier really helping, or can WBF threshold tuning alone do the same?
2) Can CLIP reduce false detections at comparable recall?
3) Can CLIP recover missed weapons from lower-confidence WBF candidates without adding too many false detections?
4) Does semantic score calibration improve the operating trade-off?

This script DOES NOT rerun YOLO detection, WBF, CLIP, or YOLO-cls inference.
It reuses saved candidate-level verifier scores from Phase 2 and Phase 3.

Inputs expected on server
-------------------------
Validation:
  phase2_valid_candidate_verifier_scores.csv
  phase1_wbf_only_valid_ground_truth.csv
  phase1_wbf_only_valid_timing_by_image.csv

Test:
  phase3_test_candidate_verifier_scores.csv
  phase1_wbf_only_valid_ground_truth.csv   (inside phase1_wbf_test folder)
  phase1_wbf_only_valid_timing_by_image.csv

Outputs
-------
CSV/XLSX tables for:
  - full validation grid
  - best validation configuration per pair/method
  - validation-selected configurations applied to test
  - WBF-only threshold curves on validation and test
  - matched-recall and matched-precision comparisons
  - mean/std summaries across detector pairs
  - focused selected-pair workbooks

Important
---------
Validation chooses configurations. Test is used only after validation selection.
WBF threshold curves on test are reported as diagnostic/curve analysis only, not for selecting settings.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

WEAPON_CLASSES = ["knife", "pistol"]
UNIFIED_CLASSES = ["knife", "pistol", "non_weapon"]
EPS = 1e-12


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_json_dump(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_float_grid(text: str) -> List[float]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


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
# Loading
# -----------------------------------------------------------------------------

def load_split_data(scores_path: Path, gt_path: Path, timing_path: Path, split_name: str):
    for p in [scores_path, gt_path, timing_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required {split_name} file: {p}")

    cand = pd.read_csv(scores_path)
    gt = pd.read_csv(gt_path)
    timing = pd.read_csv(timing_path)

    required_cand_cols = [
        "pair", "image_id", "image_path", "class_name", "score", "x1", "y1", "x2", "y2",
        "crop_valid",
        "clip_prob_detector_class", "clip_same_as_detector", "clip_entropy", "clip_inference_sec_est",
        "yolo_prob_detector_class", "yolo_same_as_detector", "yolo_entropy", "yolo_inference_sec_est",
    ]
    required_gt_cols = ["image_id", "class_name", "x1", "y1", "x2", "y2"]
    required_time_cols = ["pair", "total_sec_model1_model2_wbf"]

    missing = [c for c in required_cand_cols if c not in cand.columns]
    if missing:
        raise RuntimeError(f"Candidate score file is missing columns: {missing}\nFile: {scores_path}")
    missing = [c for c in required_gt_cols if c not in gt.columns]
    if missing:
        raise RuntimeError(f"Ground-truth file is missing columns: {missing}\nFile: {gt_path}")
    missing = [c for c in required_time_cols if c not in timing.columns]
    if missing:
        raise RuntimeError(f"Timing file is missing columns: {missing}\nFile: {timing_path}")

    cand = cand.copy()
    gt = gt.copy()
    timing = timing.copy()

    cand["class_name"] = cand["class_name"].map(canonical_name)
    gt["class_name"] = gt["class_name"].map(canonical_name)

    cand = cand[cand["class_name"].isin(WEAPON_CLASSES)].reset_index(drop=True)
    gt = gt[gt["class_name"].isin(WEAPON_CLASSES)].reset_index(drop=True)

    cand["crop_valid"] = bool_series(cand["crop_valid"])
    cand["clip_same_as_detector"] = bool_series(cand["clip_same_as_detector"])
    cand["yolo_same_as_detector"] = bool_series(cand["yolo_same_as_detector"])

    num_cols = [
        "score", "x1", "y1", "x2", "y2",
        "clip_prob_detector_class", "clip_entropy", "clip_inference_sec_est",
        "yolo_prob_detector_class", "yolo_entropy", "yolo_inference_sec_est",
    ]
    for c in num_cols:
        cand[c] = pd.to_numeric(cand[c], errors="coerce")
    for c in ["x1", "y1", "x2", "y2"]:
        gt[c] = pd.to_numeric(gt[c], errors="coerce")

    cand = cand.dropna(subset=["score", "x1", "y1", "x2", "y2"]).reset_index(drop=True)
    gt = gt.dropna(subset=["x1", "y1", "x2", "y2"]).reset_index(drop=True)

    return cand, gt, timing


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
        cls_aps = []
        for t in thresholds:
            cls_aps.append(ap_for_class(det_df, gt_df, c, float(t)))
        out[f"ap50_95_{c}"] = float(np.nanmean(cls_aps))
        all_aps.extend(cls_aps)
    out["mAP50"] = float(np.nanmean(ap50s))
    out["mAP50_95"] = float(np.nanmean(all_aps))
    return out


# -----------------------------------------------------------------------------
# Operational decision rules
# -----------------------------------------------------------------------------

def suppress_duplicates_with_kept(df: pd.DataFrame, base_keep: pd.Series, recover_mask: pd.Series, duplicate_iou: float) -> pd.Series:
    """Reject recovered boxes that overlap existing kept boxes of the same class."""
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
    """Return keep mask, verified mask, recovered mask, and score override column if needed."""
    method = cfg["method"]
    verifier = cfg.get("verifier", "none")
    base_conf = float(cfg.get("base_conf", 0.25))
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
        # No hard rejection. Candidate score is lowered when semantic agreement/probability is weak.
        # The threshold is applied to the adjusted score column.
        new_score_col = f"_{verifier}_product_score"
        df[new_score_col] = df["score"].astype(float) * prob.fillna(0.0)
        keep = (df[new_score_col] >= float(rescore_threshold)) & valid_crop
        verified = (df["score"] >= float(recovery_min_conf)) & valid_crop
        score_override = new_score_col
        return keep, verified, recovered, score_override

    if method == "semantic_rescore_blend":
        # A simple weighted blend between WBF confidence and verifier confidence.
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
    prefix = "WBF" if verifier == "none" else ("CLIP" if verifier == "clip" else "YOLO")
    if method == "WBF_threshold":
        return "WBF_threshold_curve"
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


# -----------------------------------------------------------------------------
# Config grid
# -----------------------------------------------------------------------------

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
        # WBF threshold curve. This directly addresses the fixed-threshold concern.
        for bc in wbf_grid:
            configs.append({
                "pair": pair, "method": "WBF_threshold", "verifier": "none",
                "base_conf": bc, "direct_conf": np.nan, "verifier_prob_thr": np.nan,
                "entropy_thr": np.nan, "recovery_min_conf": np.nan,
                "rescore_threshold": np.nan, "rescore_alpha": np.nan,
            })

        for verifier in ["clip", "yolo"]:
            # Filter-only methods at fixed WBF threshold.
            configs.append({
                "pair": pair, "method": "filter_hard", "verifier": verifier,
                "base_conf": args.fixed_base_conf, "direct_conf": np.nan,
                "verifier_prob_thr": np.nan, "entropy_thr": np.nan,
                "recovery_min_conf": np.nan, "rescore_threshold": np.nan, "rescore_alpha": np.nan,
            })
            for dc in direct_grid:
                for vp in prob_grid:
                    configs.append({
                        "pair": pair, "method": "filter_selective_conf", "verifier": verifier,
                        "base_conf": args.fixed_base_conf, "direct_conf": dc,
                        "verifier_prob_thr": vp, "entropy_thr": np.nan,
                        "recovery_min_conf": np.nan, "rescore_threshold": np.nan, "rescore_alpha": np.nan,
                    })
                for eh in ent_grid:
                    configs.append({
                        "pair": pair, "method": "filter_selective_entropy", "verifier": verifier,
                        "base_conf": args.fixed_base_conf, "direct_conf": dc,
                        "verifier_prob_thr": np.nan, "entropy_thr": eh,
                        "recovery_min_conf": np.nan, "rescore_threshold": np.nan, "rescore_alpha": np.nan,
                    })

            # Recovery-only: keep WBF 0.25, only add verified low-confidence boxes.
            for rmin in recovery_min_grid:
                for vp in prob_grid:
                    configs.append({
                        "pair": pair, "method": "recovery_conf", "verifier": verifier,
                        "base_conf": args.fixed_base_conf, "direct_conf": np.nan,
                        "verifier_prob_thr": vp, "entropy_thr": np.nan,
                        "recovery_min_conf": rmin, "rescore_threshold": np.nan, "rescore_alpha": np.nan,
                    })
                for eh in ent_grid:
                    configs.append({
                        "pair": pair, "method": "recovery_entropy", "verifier": verifier,
                        "base_conf": args.fixed_base_conf, "direct_conf": np.nan,
                        "verifier_prob_thr": np.nan, "entropy_thr": eh,
                        "recovery_min_conf": rmin, "rescore_threshold": np.nan, "rescore_alpha": np.nan,
                    })

            # Combined filter + recovery, entropy version only, to keep search controlled.
            for dc in direct_grid:
                for rmin in recovery_min_grid:
                    for eh in ent_grid:
                        configs.append({
                            "pair": pair, "method": "filter_recovery_entropy", "verifier": verifier,
                            "base_conf": args.fixed_base_conf, "direct_conf": dc,
                            "verifier_prob_thr": np.nan, "entropy_thr": eh,
                            "recovery_min_conf": rmin, "rescore_threshold": np.nan, "rescore_alpha": np.nan,
                        })

            # Semantic score calibration / rescoring.
            for rmin in recovery_min_grid:
                # In practice rmin defines the lowest original WBF candidate considered in this mode.
                # The saved Phase-1 candidates usually start at 0.01, but this option controls safety.
                for rt in rescore_threshold_grid:
                    configs.append({
                        "pair": pair, "method": "semantic_rescore_product", "verifier": verifier,
                        "base_conf": np.nan, "direct_conf": np.nan,
                        "verifier_prob_thr": np.nan, "entropy_thr": np.nan,
                        "recovery_min_conf": rmin, "rescore_threshold": rt, "rescore_alpha": np.nan,
                    })
                    for alpha in rescore_alpha_grid:
                        configs.append({
                            "pair": pair, "method": "semantic_rescore_blend", "verifier": verifier,
                            "base_conf": np.nan, "direct_conf": np.nan,
                            "verifier_prob_thr": np.nan, "entropy_thr": np.nan,
                            "recovery_min_conf": rmin, "rescore_threshold": rt, "rescore_alpha": alpha,
                        })
    for c in configs:
        c["method_label"] = method_label(c)
    return configs


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def pair_wbf_seconds(timing_df: pd.DataFrame) -> Dict[str, float]:
    return timing_df.groupby("pair")["total_sec_model1_model2_wbf"].sum().to_dict()


def mean_verifier_seconds(scored_df: pd.DataFrame) -> Dict[str, float]:
    valid = scored_df["crop_valid"].astype(bool)
    clip_sec = float(scored_df.loc[valid, "clip_inference_sec_est"].mean())
    yolo_sec = float(scored_df.loc[valid, "yolo_inference_sec_est"].mean())
    return {"clip": clip_sec, "yolo": yolo_sec, "none": 0.0}


def evaluate_config(scored_df: pd.DataFrame, gt_df: pd.DataFrame, timing_df: pd.DataFrame,
                    cfg: Dict, split: str, args, compute_ap: bool = False) -> Dict:
    pair = cfg["pair"]
    pair_df = scored_df[scored_df["pair"] == pair].copy()

    # For rescore modes, restrict to candidates at or above the original safety floor before calculating adjusted score.
    if cfg["method"].startswith("semantic_rescore"):
        rmin = float(cfg.get("recovery_min_conf", args.rescore_original_min_conf))
        pair_df = pair_df[pair_df["score"] >= rmin].copy()

    keep, verified, recovered, score_override = make_decision(pair_df, cfg, args.duplicate_iou)
    det = pair_df[keep].copy()
    if score_override is not None and score_override in det.columns:
        det["original_wbf_score"] = det["score"]
        det["score"] = det[score_override]

    metrics = greedy_fixed_metrics(det, gt_df, match_iou=args.match_iou)

    p1_sec = float(pair_wbf_seconds(timing_df).get(pair, 0.0))
    v_sec_map = mean_verifier_seconds(scored_df)
    n_verified = int(verified.sum())
    verifier = cfg.get("verifier", "none")
    verifier_sec = n_verified * v_sec_map.get(verifier, 0.0)
    total_sec = p1_sec + verifier_sec
    n_images = int(gt_df["image_id"].nunique())

    out = {
        "split": split,
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
        "phase1_wbf_sec": p1_sec,
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
    for cfg in tqdm(configs, desc=f"{split}: Phase-5 config grid"):
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
            "pair", "method", "method_label", "verifier", "base_conf", "direct_conf",
            "verifier_prob_thr", "entropy_thr", "recovery_min_conf", "rescore_threshold", "rescore_alpha"
        ] if k in r.index}
        rows.append(evaluate_config(scored_df, gt_df, timing_df, cfg, split, args, compute_ap=True))
    return pd.DataFrame(rows)


def mean_std_table(df: pd.DataFrame, group_col: str = "method_label") -> pd.DataFrame:
    cols = [
        "precision", "recall", "f1", "custom_fp_ratio", "fdr",
        "tp", "fp", "fn", "mAP50", "mAP50_95",
        "end_to_end_images_per_sec_est", "num_verified_candidates", "num_recovered_candidates", "num_kept_detections",
    ]
    rows = []
    for key, g in df.groupby(group_col, sort=False):
        row = {group_col: key, "num_pairs": int(g["pair"].nunique())}
        for c in cols:
            if c in g.columns:
                row[f"{c}_mean"] = float(g[c].mean())
                row[f"{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def apply_valid_selected_to_test(valid_best: pd.DataFrame, test_scored: pd.DataFrame,
                                 test_gt: pd.DataFrame, test_timing: pd.DataFrame,
                                 args) -> pd.DataFrame:
    return add_map_to_rows(valid_best, test_scored, test_gt, test_timing, "test", args, "test: frozen Phase-5 configs")


# -----------------------------------------------------------------------------
# Matched operating point analysis
# -----------------------------------------------------------------------------

def build_wbf_curve(grid_df: pd.DataFrame) -> pd.DataFrame:
    return grid_df[grid_df["method_label"] == "WBF_threshold_curve"].copy()


def matched_recall_table(method_df: pd.DataFrame, wbf_curve: pd.DataFrame, tolerance: float, split: str) -> pd.DataFrame:
    rows = []
    for _, r in method_df.iterrows():
        if r["method_label"] == "WBF_threshold_curve":
            continue
        g = wbf_curve[wbf_curve["pair"] == r["pair"]].copy()
        if len(g) == 0:
            continue
        # Prefer WBF settings with at least comparable recall; among them, use lowest FP.
        candidates = g[g["recall"] >= (float(r["recall"]) - tolerance)].copy()
        matched_type = "recall_at_or_above"
        if len(candidates) == 0:
            candidates = g.copy()
            candidates["recall_distance"] = (candidates["recall"] - float(r["recall"])).abs()
            candidates = candidates.sort_values(["recall_distance", "fp", "f1"], ascending=[True, True, False])
            matched_type = "closest_recall_no_wbf_at_or_above"
        else:
            candidates = candidates.sort_values(["fp", "recall", "f1"], ascending=[True, False, False])
        wb = candidates.iloc[0]
        rows.append({
            "split": split,
            "pair": r["pair"],
            "method_label": r["method_label"],
            "method_precision": r["precision"],
            "method_recall": r["recall"],
            "method_f1": r["f1"],
            "method_fp": int(r["fp"]),
            "method_fn": int(r["fn"]),
            "wbf_matched_conf": wb["base_conf"],
            "wbf_precision": wb["precision"],
            "wbf_recall": wb["recall"],
            "wbf_f1": wb["f1"],
            "wbf_fp": int(wb["fp"]),
            "wbf_fn": int(wb["fn"]),
            "fp_change_vs_wbf": int(r["fp"]) - int(wb["fp"]),
            "fn_change_vs_wbf": int(r["fn"]) - int(wb["fn"]),
            "f1_change_vs_wbf": float(r["f1"]) - float(wb["f1"]),
            "recall_difference": float(r["recall"]) - float(wb["recall"]),
            "matched_type": matched_type,
        })
    return pd.DataFrame(rows)


def matched_precision_table(method_df: pd.DataFrame, wbf_curve: pd.DataFrame, tolerance: float, split: str) -> pd.DataFrame:
    rows = []
    for _, r in method_df.iterrows():
        if r["method_label"] == "WBF_threshold_curve":
            continue
        g = wbf_curve[wbf_curve["pair"] == r["pair"]].copy()
        if len(g) == 0:
            continue
        # Prefer WBF settings with at least comparable precision; among them, use lowest FN.
        candidates = g[g["precision"] >= (float(r["precision"]) - tolerance)].copy()
        matched_type = "precision_at_or_above"
        if len(candidates) == 0:
            candidates = g.copy()
            candidates["precision_distance"] = (candidates["precision"] - float(r["precision"])).abs()
            candidates = candidates.sort_values(["precision_distance", "fn", "f1"], ascending=[True, True, False])
            matched_type = "closest_precision_no_wbf_at_or_above"
        else:
            candidates = candidates.sort_values(["fn", "precision", "f1"], ascending=[True, False, False])
        wb = candidates.iloc[0]
        rows.append({
            "split": split,
            "pair": r["pair"],
            "method_label": r["method_label"],
            "method_precision": r["precision"],
            "method_recall": r["recall"],
            "method_f1": r["f1"],
            "method_fp": int(r["fp"]),
            "method_fn": int(r["fn"]),
            "wbf_matched_conf": wb["base_conf"],
            "wbf_precision": wb["precision"],
            "wbf_recall": wb["recall"],
            "wbf_f1": wb["f1"],
            "wbf_fp": int(wb["fp"]),
            "wbf_fn": int(wb["fn"]),
            "fp_change_vs_wbf": int(r["fp"]) - int(wb["fp"]),
            "fn_change_vs_wbf": int(r["fn"]) - int(wb["fn"]),
            "f1_change_vs_wbf": float(r["f1"]) - float(wb["f1"]),
            "precision_difference": float(r["precision"]) - float(wb["precision"]),
            "matched_type": matched_type,
        })
    return pd.DataFrame(rows)


def candidate_final_configs(valid_best: pd.DataFrame, args) -> pd.DataFrame:
    # Keep verifier methods only, then rank using validation. WBF-only remains comparator.
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 5 operating-point diagnostics using saved Phase-2/3 scores")

    parser.add_argument("--phase1_valid_dir", type=str, default="/home/inspur/ishrai/phase1_wbf_valid")
    parser.add_argument("--phase1_test_dir", type=str, default="/home/inspur/ishrai/phase1_wbf_test")
    parser.add_argument("--phase2_valid_dir", type=str, default="/home/inspur/ishrai/phase2_wbf_verifiers_valid")
    parser.add_argument("--phase3_test_dir", type=str, default="/home/inspur/ishrai/phase3_wbf_verifiers_test")
    parser.add_argument("--out_dir", type=str, default="/home/inspur/ishrai/phase5_operating_point_diagnostics")

    parser.add_argument("--fixed_base_conf", type=float, default=0.25,
                        help="Reference WBF threshold for filter-only and recovery-only logic.")
    parser.add_argument("--wbf_threshold_grid", type=str,
                        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90",
                        help="WBF-only threshold curve grid. This directly addresses the fixed-threshold concern.")
    parser.add_argument("--direct_conf_grid", type=str, default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--verifier_prob_grid", type=str, default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--entropy_grid", type=str, default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--recovery_min_conf_grid", type=str, default="0.05,0.10,0.15,0.20",
                        help="Lowest original WBF confidence considered for recovery/rescore modes. Must be below fixed_base_conf.")
    parser.add_argument("--rescore_threshold_grid", type=str,
                        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--rescore_alpha_grid", type=str, default="0.50,0.70",
                        help="Blend weight for original WBF score in semantic_rescore_blend mode.")
    parser.add_argument("--rescore_original_min_conf", type=float, default=0.05)
    parser.add_argument("--duplicate_iou", type=float, default=0.50,
                        help="Recovery candidates overlapping an already kept same-class box above this IoU are suppressed.")
    parser.add_argument("--match_iou", type=float, default=0.50)
    parser.add_argument("--matched_tolerance", type=float, default=0.002,
                        help="Tolerance for matched-recall and matched-precision diagnostic tables.")
    parser.add_argument("--selected_pair", type=str, default="y8x+y8l")
    parser.add_argument("--limit_pairs", type=str, default="",
                        help="Optional comma-separated pair list for debug runs, e.g. y8x+y8l,y8l+y26s")
    parser.add_argument("--selection_mode", type=str, default="balanced",
                        choices=["balanced", "fp_control", "fn_recovery"],
                        help="How to choose best validation threshold within each pair/method.")
    parser.add_argument("--final_candidate_mode", type=str, default="balanced",
                        choices=["balanced", "fp_control", "fn_recovery"],
                        help="How to rank top final verifier candidates from validation.")
    parser.add_argument("--top_k_final_candidates", type=int, default=20)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    phase1_valid_dir = Path(args.phase1_valid_dir)
    phase1_test_dir = Path(args.phase1_test_dir)
    phase2_valid_dir = Path(args.phase2_valid_dir)
    phase3_test_dir = Path(args.phase3_test_dir)

    valid_scores_path = phase2_valid_dir / "phase2_valid_candidate_verifier_scores.csv"
    valid_gt_path = phase1_valid_dir / "phase1_wbf_only_valid_ground_truth.csv"
    valid_timing_path = phase1_valid_dir / "phase1_wbf_only_valid_timing_by_image.csv"

    test_scores_path = phase3_test_dir / "phase3_test_candidate_verifier_scores.csv"
    test_gt_path = phase1_test_dir / "phase1_wbf_only_valid_ground_truth.csv"
    test_timing_path = phase1_test_dir / "phase1_wbf_only_valid_timing_by_image.csv"

    print("Loading validation data...")
    valid_scored, valid_gt, valid_timing = load_split_data(valid_scores_path, valid_gt_path, valid_timing_path, "validation")
    print("Loading test data...")
    test_scored, test_gt, test_timing = load_split_data(test_scores_path, test_gt_path, test_timing_path, "test")

    pairs = sorted(valid_scored["pair"].dropna().unique().tolist())
    test_pairs = set(test_scored["pair"].dropna().unique().tolist())
    pairs = [p for p in pairs if p in test_pairs]

    if args.limit_pairs.strip():
        wanted = [x.strip() for x in args.limit_pairs.split(",") if x.strip()]
        pairs = [p for p in pairs if p in wanted]
        valid_scored = valid_scored[valid_scored["pair"].isin(pairs)].reset_index(drop=True)
        test_scored = test_scored[test_scored["pair"].isin(pairs)].reset_index(drop=True)
        valid_timing = valid_timing[valid_timing["pair"].isin(pairs)].reset_index(drop=True)
        test_timing = test_timing[test_timing["pair"].isin(pairs)].reset_index(drop=True)

    if not pairs:
        raise RuntimeError("No common detector pairs found between validation and test score files.")

    configs = build_configs(pairs, args)
    print(f"Detector pairs: {pairs}")
    print(f"Total Phase-5 validation configs: {len(configs)}")
    print(f"Fixed reference WBF base confidence: {args.fixed_base_conf}")
    print("This phase reuses saved candidates/scores; it does not rerun YOLO, WBF, CLIP, or YOLO-cls inference.")

    settings = vars(args).copy()
    settings.update({
        "note": "Phase 5 checks WBF threshold tuning, matched operating points, recovery-only, filter+recovery, and semantic score calibration.",
        "validation_scores_path": str(valid_scores_path),
        "test_scores_path": str(test_scores_path),
        "valid_num_candidates": int(len(valid_scored)),
        "test_num_candidates": int(len(test_scored)),
        "valid_num_images": int(valid_gt["image_id"].nunique()),
        "test_num_images": int(test_gt["image_id"].nunique()),
        "valid_num_gt": int(len(valid_gt)),
        "test_num_gt": int(len(test_gt)),
        "pairs": pairs,
        "detection_classes": WEAPON_CLASSES,
        "verifier_classes": UNIFIED_CLASSES,
        "test_policy": "Validation selects configurations. Test is used for frozen evaluation and threshold-curve diagnostics only, not selection.",
    })
    safe_json_dump(settings, out_dir / "phase5_operating_point_settings.json")

    # Validation full grid.
    valid_grid = evaluate_grid(valid_scored, valid_gt, valid_timing, configs, "valid", args)
    valid_grid.to_csv(out_dir / "phase5_valid_full_grid.csv", index=False)
    valid_grid.to_excel(out_dir / "phase5_valid_full_grid.xlsx", index=False)

    # Best validation config per pair/method, then mAP for those selected rows.
    valid_best_raw = select_best_by_pair_method(valid_grid, args)
    valid_best = add_map_to_rows(valid_best_raw, valid_scored, valid_gt, valid_timing, "valid", args,
                                  "valid: mAP for selected pair/method configs")
    valid_best.to_csv(out_dir / "phase5_valid_best_by_pair_method.csv", index=False)
    valid_best.to_excel(out_dir / "phase5_valid_best_by_pair_method.xlsx", index=False)

    # Apply validation-selected configs to test.
    test_frozen = apply_valid_selected_to_test(valid_best, test_scored, test_gt, test_timing, args)
    test_frozen.to_csv(out_dir / "phase5_test_frozen_results.csv", index=False)
    test_frozen.to_excel(out_dir / "phase5_test_frozen_results.xlsx", index=False)

    # WBF threshold curve on test, for diagnostic/matched operating point analysis only.
    wbf_test_configs = [c for c in configs if c["method_label"] == "WBF_threshold_curve"]
    test_wbf_curve = evaluate_grid(test_scored, test_gt, test_timing, wbf_test_configs, "test", args)
    test_wbf_curve.to_csv(out_dir / "phase5_test_wbf_threshold_curve.csv", index=False)
    test_wbf_curve.to_excel(out_dir / "phase5_test_wbf_threshold_curve.xlsx", index=False)

    # Summaries across pairs.
    valid_meanstd = mean_std_table(valid_best)
    test_meanstd = mean_std_table(test_frozen)
    valid_meanstd.to_csv(out_dir / "phase5_valid_mean_std_across_pairs.csv", index=False)
    valid_meanstd.to_excel(out_dir / "phase5_valid_mean_std_across_pairs.xlsx", index=False)
    test_meanstd.to_csv(out_dir / "phase5_test_mean_std_across_pairs.csv", index=False)
    test_meanstd.to_excel(out_dir / "phase5_test_mean_std_across_pairs.xlsx", index=False)

    # Matched operating point diagnostics.
    valid_wbf_curve = build_wbf_curve(valid_grid)
    valid_matched_recall = matched_recall_table(valid_best, valid_wbf_curve, args.matched_tolerance, "valid")
    valid_matched_precision = matched_precision_table(valid_best, valid_wbf_curve, args.matched_tolerance, "valid")
    test_matched_recall = matched_recall_table(test_frozen, test_wbf_curve, args.matched_tolerance, "test")
    test_matched_precision = matched_precision_table(test_frozen, test_wbf_curve, args.matched_tolerance, "test")

    valid_matched_recall.to_csv(out_dir / "phase5_valid_matched_recall_vs_wbf.csv", index=False)
    valid_matched_precision.to_csv(out_dir / "phase5_valid_matched_precision_vs_wbf.csv", index=False)
    test_matched_recall.to_csv(out_dir / "phase5_test_matched_recall_vs_wbf.csv", index=False)
    test_matched_precision.to_csv(out_dir / "phase5_test_matched_precision_vs_wbf.csv", index=False)
    valid_matched_recall.to_excel(out_dir / "phase5_valid_matched_recall_vs_wbf.xlsx", index=False)
    valid_matched_precision.to_excel(out_dir / "phase5_valid_matched_precision_vs_wbf.xlsx", index=False)
    test_matched_recall.to_excel(out_dir / "phase5_test_matched_recall_vs_wbf.xlsx", index=False)
    test_matched_precision.to_excel(out_dir / "phase5_test_matched_precision_vs_wbf.xlsx", index=False)

    # Top final candidates from validation and their test results.
    top_valid_candidates = candidate_final_configs(valid_best, args)
    key_cols = [
        "pair", "method_label", "method", "verifier", "base_conf", "direct_conf", "verifier_prob_thr",
        "entropy_thr", "recovery_min_conf", "rescore_threshold", "rescore_alpha"
    ]
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

    top_valid_candidates.to_csv(out_dir / "phase5_top_validation_final_candidates.csv", index=False)
    top_test_results.to_csv(out_dir / "phase5_top_validation_candidates_on_test.csv", index=False)
    selected_pair_valid.to_csv(out_dir / "phase5_selected_pair_valid_results.csv", index=False)
    selected_pair_test.to_csv(out_dir / "phase5_selected_pair_test_results.csv", index=False)
    top_valid_candidates.to_excel(out_dir / "phase5_top_validation_final_candidates.xlsx", index=False)
    top_test_results.to_excel(out_dir / "phase5_top_validation_candidates_on_test.xlsx", index=False)
    selected_pair_valid.to_excel(out_dir / "phase5_selected_pair_valid_results.xlsx", index=False)
    selected_pair_test.to_excel(out_dir / "phase5_selected_pair_test_results.xlsx", index=False)

    # Compact paper summary workbook.
    paper_xlsx = out_dir / "phase5_paper_decision_summary.xlsx"
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
        valid_wbf_curve.to_excel(writer, sheet_name="valid_wbf_curve", index=False)
        test_wbf_curve.to_excel(writer, sheet_name="test_wbf_curve", index=False)
        pd.DataFrame([settings]).to_excel(writer, sheet_name="settings", index=False)

    print("\n" + "=" * 96)
    print("PHASE 5 OPERATING-POINT DIAGNOSTICS COMPLETE")
    print(f"Output folder                         : {out_dir}")
    print(f"Full validation grid                  : {out_dir / 'phase5_valid_full_grid.xlsx'}")
    print(f"Best validation by pair/method        : {out_dir / 'phase5_valid_best_by_pair_method.xlsx'}")
    print(f"Frozen test results                   : {out_dir / 'phase5_test_frozen_results.xlsx'}")
    print(f"Test WBF threshold curve              : {out_dir / 'phase5_test_wbf_threshold_curve.xlsx'}")
    print(f"Matched recall diagnostics            : {out_dir / 'phase5_test_matched_recall_vs_wbf.xlsx'}")
    print(f"Matched precision diagnostics         : {out_dir / 'phase5_test_matched_precision_vs_wbf.xlsx'}")
    print(f"Paper decision summary                : {paper_xlsx}")
    print(f"Settings JSON                         : {out_dir / 'phase5_operating_point_settings.json'}")
    print("=" * 96)

    show_cols = [
        "split", "pair", "method_label", "precision", "recall", "f1", "fp", "fn",
        "custom_fp_ratio", "base_conf", "direct_conf", "verifier_prob_thr", "entropy_thr",
        "recovery_min_conf", "rescore_threshold", "rescore_alpha", "mAP50", "mAP50_95",
        "num_recovered_candidates", "end_to_end_images_per_sec_est",
    ]
    show_cols = [c for c in show_cols if c in top_valid_candidates.columns]
    print("\nTop validation-selected verifier candidates:")
    if len(top_valid_candidates):
        print(top_valid_candidates[show_cols].head(15).to_string(index=False))
    else:
        print("No verifier candidates found.")

    show_cols_test = [c for c in show_cols if c in top_test_results.columns]
    print("\nSame top validation candidates applied to test:")
    if len(top_test_results):
        print(top_test_results[show_cols_test].head(15).to_string(index=False))
    else:
        print("No top candidate test rows found.")


if __name__ == "__main__":
    main()
