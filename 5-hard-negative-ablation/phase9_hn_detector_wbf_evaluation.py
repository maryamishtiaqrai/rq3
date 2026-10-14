#!/usr/bin/env python3
"""
Phase 9: Evaluate hard-negative fine-tuned YOLO detectors and WBF ensembles.

Purpose
-------
This script evaluates the new D5_HN fine-tuned detector weights:
  y8x_best.pt, y8l_best.pt, y26n_best.pt, y26s_best.pt

It does NOT use CLIP or YOLO-cls verification. This phase checks whether
hard-negative fine-tuning improved the detector/WBF baseline itself.

Main outputs
------------
- Single-detector threshold curves on valid/test
- WBF threshold curves on valid/test
- Validation-selected thresholds applied to test
- Candidate CSVs saved for later verifier phases
- Timing and throughput summaries

Detector classes remain:
  0 = knife
  1 = pistol
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = {0: "knife", 1: "pistol"}
MODEL_FILES = {
    "y8x": "y8x_best.pt",
    "y8l": "y8l_best.pt",
    "y26n": "y26n_best.pt",
    "y26s": "y26s_best.pt",
}
DEFAULT_PAIRS = [
    ("y8x", "y8l"),
    ("y8x", "y26n"),
    ("y8x", "y26s"),
    ("y8l", "y26n"),
    ("y8l", "y26s"),
    ("y26n", "y26s"),
]


@dataclass
class ImageInfo:
    image_id: str
    image_path: str
    label_path: str
    width: int
    height: int


def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_strings(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def resolve_split_path(yaml_path: str, split: str) -> Path:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = Path(data.get("path", Path(yaml_path).parent)).expanduser()
    key = "val" if split in {"valid", "val"} else split
    if key not in data:
        raise KeyError(f"Split '{split}' not found in yaml. Available keys: {list(data.keys())}")
    split_value = data[key]
    if isinstance(split_value, list):
        # Keep only the first path if yaml contains list. Most YOLO datasets use a string.
        split_value = split_value[0]
    p = Path(split_value).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def label_path_from_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    # fallback
    return image_path.with_suffix(".txt")


def list_images(split_dir: Path, limit: Optional[int] = None) -> List[Path]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split image directory not found: {split_dir}")
    imgs = [p for p in split_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    imgs = sorted(imgs)
    if limit is not None and limit > 0:
        imgs = imgs[:limit]
    return imgs


def load_image_infos(yaml_path: str, split: str, limit: Optional[int] = None) -> List[ImageInfo]:
    split_dir = resolve_split_path(yaml_path, split)
    imgs = list_images(split_dir, limit)
    infos = []
    for img_path in tqdm(imgs, desc=f"Reading image sizes [{split}]", leave=False):
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"WARNING: could not read image, skipping: {img_path}")
            continue
        h, w = im.shape[:2]
        infos.append(ImageInfo(
            image_id=str(img_path),
            image_path=str(img_path),
            label_path=str(label_path_from_image(img_path)),
            width=int(w),
            height=int(h),
        ))
    return infos


def load_ground_truth(infos: List[ImageInfo]) -> pd.DataFrame:
    rows = []
    for info in infos:
        lp = Path(info.label_path)
        if not lp.exists():
            continue
        text = lp.read_text(errors="ignore").strip()
        if not text:
            continue
        for line_no, line in enumerate(text.splitlines()):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
            except Exception:
                continue
            x1 = (xc - bw / 2.0) * info.width
            y1 = (yc - bh / 2.0) * info.height
            x2 = (xc + bw / 2.0) * info.width
            y2 = (yc + bh / 2.0) * info.height
            rows.append({
                "image_id": info.image_id,
                "image_path": info.image_path,
                "label_path": info.label_path,
                "width": info.width,
                "height": info.height,
                "gt_id": f"{info.image_id}__gt{line_no}",
                "class_id": cls,
                "class_name": CLASS_NAMES.get(cls, str(cls)),
                "x1": max(0.0, min(float(x1), info.width - 1)),
                "y1": max(0.0, min(float(y1), info.height - 1)),
                "x2": max(0.0, min(float(x2), info.width - 1)),
                "y2": max(0.0, min(float(y2), info.height - 1)),
            })
    return pd.DataFrame(rows)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else float(inter / denom)


def box_iou_matrix(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    denom = area_a + area_b - inter
    return np.where(denom > 0, inter / denom, 0.0)


def run_detector(model_id: str, weight_path: Path, infos: List[ImageInfo], args) -> Tuple[pd.DataFrame, Dict]:
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing detector weight for {model_id}: {weight_path}")
    print(f"\nLoading detector {model_id}: {weight_path}")
    model = YOLO(str(weight_path))
    rows = []
    total_pred_time = 0.0
    n_images = 0

    for info in tqdm(infos, desc=f"Predicting {model_id}"):
        t0 = time.perf_counter()
        res = model.predict(
            source=info.image_path,
            imgsz=args.imgsz,
            conf=args.low_conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )[0]
        dt = time.perf_counter() - t0
        total_pred_time += dt
        n_images += 1
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.detach().cpu().numpy()
        conf = res.boxes.conf.detach().cpu().numpy()
        cls = res.boxes.cls.detach().cpu().numpy().astype(int)
        for j in range(len(conf)):
            c = int(cls[j])
            if c not in CLASS_NAMES:
                continue
            rows.append({
                "source_type": "single_detector",
                "model_set": "HN",
                "entity": model_id,
                "detector": model_id,
                "image_id": info.image_id,
                "image_path": info.image_path,
                "width": info.width,
                "height": info.height,
                "class_id": c,
                "class_name": CLASS_NAMES[c],
                "conf": float(conf[j]),
                "x1": float(xyxy[j, 0]),
                "y1": float(xyxy[j, 1]),
                "x2": float(xyxy[j, 2]),
                "y2": float(xyxy[j, 3]),
                "num_support_models": 1,
                "support_models": model_id,
            })
    timing = {
        "model_set": "HN",
        "source_type": "single_detector",
        "entity": model_id,
        "n_images": n_images,
        "total_detector_sec": total_pred_time,
        "wbf_sec": 0.0,
        "total_pipeline_sec_est": total_pred_time,
        "images_per_sec_est": n_images / total_pred_time if total_pred_time > 0 else np.nan,
        "ms_per_image_est": 1000.0 * total_pred_time / n_images if n_images > 0 else np.nan,
    }
    return pd.DataFrame(rows), timing


def weighted_boxes_fusion_for_image(preds: pd.DataFrame, model_ids: Tuple[str, ...], iou_thr: float) -> List[Dict]:
    out = []
    if preds.empty:
        return out
    n_models = len(model_ids)
    for cls_id, cdf in preds.groupby("class_id"):
        boxes = cdf[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        scores = cdf["conf"].to_numpy(dtype=float)
        models = cdf["detector"].astype(str).to_numpy()
        order = np.argsort(-scores)
        used = np.zeros(len(order), dtype=bool)
        for ii, ord_idx in enumerate(order):
            if used[ii]:
                continue
            seed_box = boxes[ord_idx]
            cluster_order_positions = [ii]
            used[ii] = True
            # Greedy cluster against current weighted box.
            cluster_indices = [ord_idx]
            cluster_scores = [scores[ord_idx]]
            current_box = seed_box.copy()
            for jj in range(ii + 1, len(order)):
                if used[jj]:
                    continue
                idx = order[jj]
                if iou_xyxy(current_box, boxes[idx]) >= iou_thr:
                    used[jj] = True
                    cluster_order_positions.append(jj)
                    cluster_indices.append(idx)
                    cluster_scores.append(scores[idx])
                    w = np.asarray(cluster_scores, dtype=float)
                    b = boxes[np.asarray(cluster_indices)]
                    current_box = np.average(b, axis=0, weights=np.maximum(w, 1e-9))
            cluster_indices_arr = np.asarray(cluster_indices)
            cluster_scores_arr = scores[cluster_indices_arr]
            weights = np.maximum(cluster_scores_arr, 1e-9)
            fused_box = np.average(boxes[cluster_indices_arr], axis=0, weights=weights)
            unique_models = sorted(set(models[cluster_indices_arr].tolist()))
            raw_score = float(np.mean(cluster_scores_arr))
            # Same logic as conf_rescale=min with equal weights: support fraction.
            fused_score = raw_score * (len(unique_models) / float(n_models))
            out.append({
                "class_id": int(cls_id),
                "class_name": CLASS_NAMES.get(int(cls_id), str(cls_id)),
                "conf": float(fused_score),
                "raw_conf_mean": raw_score,
                "x1": float(fused_box[0]),
                "y1": float(fused_box[1]),
                "x2": float(fused_box[2]),
                "y2": float(fused_box[3]),
                "num_support_models": int(len(unique_models)),
                "support_models": "+".join(unique_models),
            })
    return out


def build_wbf_candidates(split: str, infos: List[ImageInfo], det_preds: Dict[str, pd.DataFrame], det_timings: Dict[str, Dict], pairs: List[Tuple[str, ...]], args) -> Tuple[pd.DataFrame, List[Dict]]:
    rows = []
    timings = []
    for pair in tqdm(pairs, desc=f"Building WBF candidates [{split}]"):
        entity = "+".join(pair)
        pair_frames = []
        for mid in pair:
            if mid not in det_preds:
                raise KeyError(f"Missing detector predictions for {mid}")
            pair_frames.append(det_preds[mid])
        all_pair_preds = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
        t0 = time.perf_counter()
        for info in infos:
            img_pred = all_pair_preds[all_pair_preds["image_id"] == info.image_id]
            fused = weighted_boxes_fusion_for_image(img_pred, pair, args.wbf_iou)
            for k, r in enumerate(fused):
                if r["conf"] < args.save_conf:
                    continue
                rows.append({
                    "source_type": "wbf",
                    "model_set": "HN",
                    "entity": entity,
                    "detector": entity,
                    "image_id": info.image_id,
                    "image_path": info.image_path,
                    "width": info.width,
                    "height": info.height,
                    "class_id": r["class_id"],
                    "class_name": r["class_name"],
                    "conf": r["conf"],
                    "raw_conf_mean": r["raw_conf_mean"],
                    "x1": r["x1"],
                    "y1": r["y1"],
                    "x2": r["x2"],
                    "y2": r["y2"],
                    "num_support_models": r["num_support_models"],
                    "support_models": r["support_models"],
                })
        wbf_sec = time.perf_counter() - t0
        det_sec = sum(det_timings[mid]["total_detector_sec"] for mid in pair)
        total_sec = det_sec + wbf_sec
        n_images = len(infos)
        timings.append({
            "model_set": "HN",
            "source_type": "wbf",
            "entity": entity,
            "n_images": n_images,
            "total_detector_sec": det_sec,
            "wbf_sec": wbf_sec,
            "total_pipeline_sec_est": total_sec,
            "images_per_sec_est": n_images / total_sec if total_sec > 0 else np.nan,
            "ms_per_image_est": 1000.0 * total_sec / n_images if n_images > 0 else np.nan,
        })
    return pd.DataFrame(rows), timings


def evaluate_fixed(preds: pd.DataFrame, gt: pd.DataFrame, threshold: float, match_iou: float) -> Dict:
    if preds.empty:
        preds_f = preds.copy()
    else:
        preds_f = preds[preds["conf"] >= threshold].copy()
    return evaluate_predictions(preds_f, gt, match_iou)


def evaluate_predictions(preds: pd.DataFrame, gt: pd.DataFrame, match_iou: float) -> Dict:
    # Convert empty dataframes safely.
    gt_by_image = {k: v for k, v in gt.groupby("image_id")} if not gt.empty else {}
    pred_by_image = {k: v for k, v in preds.groupby("image_id")} if not preds.empty else {}
    image_ids = sorted(set(gt_by_image.keys()) | set(pred_by_image.keys()))

    tp = 0
    fp = 0
    fn = 0
    per_class = {0: {"tp": 0, "fp": 0, "fn": 0}, 1: {"tp": 0, "fp": 0, "fn": 0}}

    for image_id in image_ids:
        g = gt_by_image.get(image_id, pd.DataFrame(columns=gt.columns))
        p = pred_by_image.get(image_id, pd.DataFrame(columns=preds.columns if not preds.empty else ["conf", "class_id", "x1", "y1", "x2", "y2"]))
        if not p.empty:
            p = p.sort_values("conf", ascending=False)
        matched_gt_indices = set()
        for _, pr in p.iterrows():
            cls = int(pr["class_id"])
            pbox = np.array([pr["x1"], pr["y1"], pr["x2"], pr["y2"]], dtype=float)
            g_cls = g[g["class_id"].astype(int) == cls]
            best_iou = 0.0
            best_idx = None
            for gi, gr in g_cls.iterrows():
                if gi in matched_gt_indices:
                    continue
                gbox = np.array([gr["x1"], gr["y1"], gr["x2"], gr["y2"]], dtype=float)
                iou = iou_xyxy(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_idx is not None and best_iou >= match_iou:
                tp += 1
                per_class[cls]["tp"] += 1
                matched_gt_indices.add(best_idx)
            else:
                fp += 1
                if cls in per_class:
                    per_class[cls]["fp"] += 1
        # all unmatched GT are FN
        for gi, gr in g.iterrows():
            if gi not in matched_gt_indices:
                cls = int(gr["class_id"])
                fn += 1
                if cls in per_class:
                    per_class[cls]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0

    out = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "custom_fp_ratio": fdr,
        "num_gt": int(len(gt)),
        "num_predictions": int(len(preds)),
    }
    for cls, d in per_class.items():
        ctp, cfp, cfn = d["tp"], d["fp"], d["fn"]
        cp = ctp / (ctp + cfp) if (ctp + cfp) else 0.0
        cr = ctp / (ctp + cfn) if (ctp + cfn) else 0.0
        cf1 = 2 * cp * cr / (cp + cr) if (cp + cr) else 0.0
        name = CLASS_NAMES[cls]
        out[f"{name}_precision"] = cp
        out[f"{name}_recall"] = cr
        out[f"{name}_f1"] = cf1
        out[f"{name}_tp"] = int(ctp)
        out[f"{name}_fp"] = int(cfp)
        out[f"{name}_fn"] = int(cfn)
    return out


def ap_for_class(preds_c: pd.DataFrame, gt_c: pd.DataFrame, iou_thr: float) -> float:
    npos = len(gt_c)
    if npos == 0:
        return np.nan
    if preds_c.empty:
        return 0.0
    gt_by_image = {k: v for k, v in gt_c.groupby("image_id")}
    matched = {image_id: set() for image_id in gt_by_image}
    preds_sorted = preds_c.sort_values("conf", ascending=False)
    tp = np.zeros(len(preds_sorted), dtype=float)
    fp = np.zeros(len(preds_sorted), dtype=float)
    for idx, (_, pr) in enumerate(preds_sorted.iterrows()):
        image_id = pr["image_id"]
        g = gt_by_image.get(image_id)
        if g is None or g.empty:
            fp[idx] = 1.0
            continue
        pbox = np.array([pr["x1"], pr["y1"], pr["x2"], pr["y2"]], dtype=float)
        best_iou = 0.0
        best_gt = None
        for gi, gr in g.iterrows():
            if gi in matched[image_id]:
                continue
            gbox = np.array([gr["x1"], gr["y1"], gr["x2"], gr["y2"]], dtype=float)
            iou = iou_xyxy(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_gt is not None and best_iou >= iou_thr:
            tp[idx] = 1.0
            matched[image_id].add(best_gt)
        else:
            fp[idx] = 1.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(npos, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # 101-point interpolated AP.
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        prec_at_r = precisions[recalls >= r]
        ap += (np.max(prec_at_r) if prec_at_r.size else 0.0) / 101.0
    return float(ap)


def compute_map(preds: pd.DataFrame, gt: pd.DataFrame) -> Tuple[float, float]:
    if preds.empty:
        return 0.0, 0.0
    ap50s = []
    ap5095s = []
    for cls in sorted(CLASS_NAMES):
        preds_c = preds[preds["class_id"].astype(int) == cls]
        gt_c = gt[gt["class_id"].astype(int) == cls]
        ap50 = ap_for_class(preds_c, gt_c, 0.50)
        aps = [ap_for_class(preds_c, gt_c, thr) for thr in np.arange(0.50, 0.96, 0.05)]
        ap50s.append(ap50)
        ap5095s.append(np.nanmean(aps))
    return float(np.nanmean(ap50s)), float(np.nanmean(ap5095s))


def evaluate_threshold_curve(preds: pd.DataFrame, gt: pd.DataFrame, thresholds: List[float], match_iou: float, meta: Dict, timing_lookup: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    entity = meta["entity"]
    timing = timing_lookup.get(entity, {})
    for thr in thresholds:
        p_thr = preds[preds["conf"] >= thr].copy() if not preds.empty else preds.copy()
        metrics = evaluate_predictions(p_thr, gt, match_iou)
        ap50, ap5095 = compute_map(p_thr, gt)
        row = dict(meta)
        row.update({
            "conf_threshold": float(thr),
            "mAP50": ap50,
            "mAP50_95": ap5095,
        })
        row.update(metrics)
        for k in ["n_images", "total_detector_sec", "wbf_sec", "total_pipeline_sec_est", "images_per_sec_est", "ms_per_image_est"]:
            row[k] = timing.get(k, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_from_valid(curve_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in curve_df.groupby(["source_type", "entity"]):
        gg = g.sort_values(
            by=["f1", "fp", "fn", "precision", "recall"],
            ascending=[False, True, True, False, False],
        )
        rows.append(gg.iloc[0].to_dict())
    return pd.DataFrame(rows)


def apply_valid_thresholds_to_test(best_valid: pd.DataFrame, test_curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, br in best_valid.iterrows():
        source_type = br["source_type"]
        entity = br["entity"]
        thr = float(br["conf_threshold"])
        candidates = test_curve[(test_curve["source_type"] == source_type) & (test_curve["entity"] == entity)]
        if candidates.empty:
            continue
        candidates = candidates.copy()
        candidates["thr_diff"] = (candidates["conf_threshold"] - thr).abs()
        row = candidates.sort_values("thr_diff").iloc[0].to_dict()
        row["selected_from_valid_conf_threshold"] = thr
        row["valid_f1_for_selected_threshold"] = br["f1"]
        row["valid_precision_for_selected_threshold"] = br["precision"]
        row["valid_recall_for_selected_threshold"] = br["recall"]
        row["valid_fp_for_selected_threshold"] = br["fp"]
        row["valid_fn_for_selected_threshold"] = br["fn"]
        rows.append(row)
    return pd.DataFrame(rows)


def mean_std_by_source(curve_best: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["precision", "recall", "f1", "fp", "fn", "custom_fp_ratio", "mAP50", "mAP50_95", "images_per_sec_est"]
    rows = []
    for st, g in curve_best.groupby("source_type"):
        row = {"source_type": st, "n_entities": len(g)}
        for col in metric_cols:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_std"] = g[col].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def save_xlsx(path: Path, sheets: Dict[str, pd.DataFrame]):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)


def main():
    ap = argparse.ArgumentParser("Phase 9 HN detector and WBF evaluation")
    ap.add_argument("--data_yaml", required=True)
    ap.add_argument("--hn_model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--detectors", type=str, default="y8x,y8l,y26n,y26s")
    ap.add_argument("--splits", type=str, default="valid,test")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--low_conf", type=float, default=0.001)
    ap.add_argument("--nms_iou", type=float, default=0.70)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--wbf_iou", type=float, default=0.55)
    ap.add_argument("--save_conf", type=float, default=0.01)
    ap.add_argument("--fixed_conf", type=float, default=0.25)
    ap.add_argument("--match_iou", type=float, default=0.50)
    ap.add_argument("--threshold_grid", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90")
    ap.add_argument("--limit", type=int, default=0, help="Optional image limit per split for debugging")
    ap.add_argument("--reuse_candidates", action="store_true", help="Reuse saved candidate CSVs if present")
    ap.add_argument("--skip_wbf", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    detectors = parse_csv_strings(args.detectors)
    splits = parse_csv_strings(args.splits)
    thresholds = parse_csv_floats(args.threshold_grid)
    if args.fixed_conf not in thresholds:
        thresholds = sorted(set(thresholds + [args.fixed_conf]))
    limit = args.limit if args.limit and args.limit > 0 else None

    # Build WBF pairs from the requested detectors.
    pairs = [tuple(p) for p in DEFAULT_PAIRS if all(m in detectors for m in p)]

    all_curves = []
    all_best_valid = []
    all_test_frozen = []
    summary_sheets = {}
    settings = vars(args).copy()
    settings.update({
        "detector_classes": CLASS_NAMES,
        "wbf_pairs": ["+".join(p) for p in pairs],
        "hard_negative_note": "Evaluation of D5_HN fine-tuned detector models. Verifiers are not used in Phase 9.",
    })

    per_split_curves = {}
    per_split_timings = {}
    per_split_gt = {}

    for split in splits:
        print("\n" + "=" * 80)
        print(f"PHASE 9 SPLIT: {split}")
        print("=" * 80)
        split_tag = "valid" if split in {"val", "valid"} else split
        infos = load_image_infos(args.data_yaml, split, limit=limit)
        gt = load_ground_truth(infos)
        gt_path = out_dir / f"phase9_{split_tag}_ground_truth.csv"
        gt.to_csv(gt_path, index=False)
        per_split_gt[split_tag] = gt
        print(f"Images: {len(infos)} | GT boxes: {len(gt)} | Saved GT: {gt_path}")

        single_candidate_path = out_dir / f"phase9_{split_tag}_HN_single_detector_candidates.csv"
        wbf_candidate_path = out_dir / f"phase9_{split_tag}_HN_wbf_candidates.csv"
        timing_path = out_dir / f"phase9_{split_tag}_HN_timing.csv"

        det_preds: Dict[str, pd.DataFrame] = {}
        det_timings: Dict[str, Dict] = {}
        timing_rows = []

        if args.reuse_candidates and single_candidate_path.exists():
            single_all = pd.read_csv(single_candidate_path)
            print(f"Reused single candidates: {single_candidate_path}")
            # Timings may be unavailable when reused; try to read timing csv.
            if timing_path.exists():
                tdf = pd.read_csv(timing_path)
                for _, r in tdf[tdf["source_type"] == "single_detector"].iterrows():
                    det_timings[str(r["entity"])] = r.to_dict()
            for mid in detectors:
                det_preds[mid] = single_all[single_all["entity"] == mid].copy()
        else:
            single_frames = []
            for mid in detectors:
                wf = Path(args.hn_model_dir) / MODEL_FILES[mid]
                df, timing = run_detector(mid, wf, infos, args)
                det_preds[mid] = df
                det_timings[mid] = timing
                timing_rows.append(timing)
                single_frames.append(df)
            single_all = pd.concat(single_frames, ignore_index=True) if single_frames else pd.DataFrame()
            single_all.to_csv(single_candidate_path, index=False)
            print(f"Saved single-detector candidates: {single_candidate_path}")

        if not args.skip_wbf:
            if args.reuse_candidates and wbf_candidate_path.exists():
                wbf_all = pd.read_csv(wbf_candidate_path)
                print(f"Reused WBF candidates: {wbf_candidate_path}")
                if timing_path.exists():
                    tdf = pd.read_csv(timing_path)
                    timing_rows = tdf.to_dict("records")
            else:
                wbf_all, wbf_timings = build_wbf_candidates(split_tag, infos, det_preds, det_timings, pairs, args)
                wbf_all.to_csv(wbf_candidate_path, index=False)
                timing_rows.extend(wbf_timings)
                print(f"Saved WBF candidates: {wbf_candidate_path}")
        else:
            wbf_all = pd.DataFrame()

        # Make sure timing includes single rows when reused.
        if timing_path.exists() and args.reuse_candidates:
            timing_df_existing = pd.read_csv(timing_path)
            timing_lookup = {str(r["entity"]): r.to_dict() for _, r in timing_df_existing.iterrows()}
        else:
            timing_df = pd.DataFrame(timing_rows)
            timing_df.to_csv(timing_path, index=False)
            timing_lookup = {str(r["entity"]): r for r in timing_rows}
            print(f"Saved timing: {timing_path}")

        # Evaluate threshold curves.
        curve_frames = []
        for mid in detectors:
            p = single_all[single_all["entity"] == mid].copy() if not single_all.empty else pd.DataFrame()
            meta = {"split": split_tag, "model_set": "HN", "source_type": "single_detector", "entity": mid}
            curve_frames.append(evaluate_threshold_curve(p, gt, thresholds, args.match_iou, meta, timing_lookup))
        if not args.skip_wbf and not wbf_all.empty:
            for pair in pairs:
                ent = "+".join(pair)
                p = wbf_all[wbf_all["entity"] == ent].copy()
                meta = {"split": split_tag, "model_set": "HN", "source_type": "wbf", "entity": ent}
                curve_frames.append(evaluate_threshold_curve(p, gt, thresholds, args.match_iou, meta, timing_lookup))
        curve_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
        curve_csv = out_dir / f"phase9_{split_tag}_HN_threshold_curve.csv"
        curve_xlsx = out_dir / f"phase9_{split_tag}_HN_threshold_curve.xlsx"
        curve_df.to_csv(curve_csv, index=False)
        save_xlsx(curve_xlsx, {
            "threshold_curve": curve_df,
            "timing": pd.DataFrame(list(timing_lookup.values())),
        })
        print(f"Saved threshold curve: {curve_xlsx}")
        per_split_curves[split_tag] = curve_df
        per_split_timings[split_tag] = pd.DataFrame(list(timing_lookup.values()))

    # Validation selection and test frozen evaluation.
    valid_curve = per_split_curves.get("valid") or per_split_curves.get("val")
    test_curve = per_split_curves.get("test")
    if valid_curve is not None and not valid_curve.empty:
        best_valid = select_best_from_valid(valid_curve)
        best_valid_path = out_dir / "phase9_HN_valid_best_by_entity.xlsx"
        best_valid_csv = out_dir / "phase9_HN_valid_best_by_entity.csv"
        best_valid.to_csv(best_valid_csv, index=False)
        fixed_valid = valid_curve[np.isclose(valid_curve["conf_threshold"], args.fixed_conf)].copy()
        save_xlsx(best_valid_path, {
            "best_valid": best_valid,
            "fixed_conf_valid": fixed_valid,
            "mean_std_best_valid": mean_std_by_source(best_valid),
            "mean_std_fixed_valid": mean_std_by_source(fixed_valid),
        })
        print(f"Saved valid best summary: {best_valid_path}")
        summary_sheets["valid_best"] = best_valid
        summary_sheets["valid_fixed_conf"] = fixed_valid
        summary_sheets["valid_mean_std_best"] = mean_std_by_source(best_valid)
        summary_sheets["valid_mean_std_fixed"] = mean_std_by_source(fixed_valid)

        if test_curve is not None and not test_curve.empty:
            test_frozen = apply_valid_thresholds_to_test(best_valid, test_curve)
            fixed_test = test_curve[np.isclose(test_curve["conf_threshold"], args.fixed_conf)].copy()
            test_frozen_path = out_dir / "phase9_HN_test_frozen_from_valid.xlsx"
            test_frozen_csv = out_dir / "phase9_HN_test_frozen_from_valid.csv"
            test_frozen.to_csv(test_frozen_csv, index=False)
            save_xlsx(test_frozen_path, {
                "test_frozen_from_valid": test_frozen,
                "fixed_conf_test": fixed_test,
                "mean_std_test_frozen": mean_std_by_source(test_frozen),
                "mean_std_fixed_test": mean_std_by_source(fixed_test),
            })
            print(f"Saved test frozen summary: {test_frozen_path}")
            summary_sheets["test_frozen_from_valid"] = test_frozen
            summary_sheets["test_fixed_conf"] = fixed_test
            summary_sheets["test_mean_std_frozen"] = mean_std_by_source(test_frozen)
            summary_sheets["test_mean_std_fixed"] = mean_std_by_source(fixed_test)

    # Save paper summary workbook.
    settings_path = out_dir / "phase9_HN_detector_wbf_settings.json"
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    if summary_sheets:
        paper_summary = out_dir / "phase9_HN_detector_wbf_paper_summary.xlsx"
        save_xlsx(paper_summary, summary_sheets)
        print(f"Saved paper summary workbook: {paper_summary}")

    print("\n" + "=" * 80)
    print("PHASE 9 COMPLETE")
    print(f"Output folder: {out_dir}")
    print(f"Settings JSON: {settings_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
