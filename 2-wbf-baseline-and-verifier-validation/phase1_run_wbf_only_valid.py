#!/usr/bin/env python3
"""
Phase 1: WBF-only validation baselines for six detector pairs.

Purpose
-------
1. Run the four frozen YOLO detection models on the D5-clean validation split.
2. Fuse each of the six detector pairs using WBF.
3. Save all fused candidates once, so later phases can reuse boxes without fresh detector inference.
4. Evaluate WBF-only baselines at a fixed operating confidence threshold.
5. Record timing and throughput: images/second and ms/image for every pair.

Default model locations
-----------------------
/home/inspur/ishrai/all_togather/y8x_best.pt
/home/inspur/ishrai/all_togather/y8l_best.pt
/home/inspur/ishrai/all_togather/y26n_best.pt
/home/inspur/ishrai/all_togather/y26s_best.pt

Default outputs
---------------
/home/inspur/ishrai/phase1_wbf_valid/phase1_wbf_only_valid_summary.xlsx
/home/inspur/ishrai/phase1_wbf_valid/phase1_wbf_only_valid_all_fused_candidates.csv
/home/inspur/ishrai/phase1_wbf_valid/phase1_wbf_only_valid_timing_by_image.csv

Notes
-----
- WBF settings follow the previously used controlled protocol:
  low_conf=0.001, WBF IoU=0.55, equal model weights, confidence type=avg,
  confidence rescaling=min, fixed WBF-only evaluation confidence=0.25.
- All fused candidates are saved down to save_conf=0.01 by default. This keeps enough
  candidates for later threshold studies while avoiding extremely large Excel files.
- This script evaluates only WBF-only. It does not apply CLIP or YOLO-cls verification.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_NAMES = {0: "knife", 1: "pistol"}


@dataclass
class BoxItem:
    box: np.ndarray       # normalized xyxy, shape (4,)
    score: float
    label: int
    model_idx: int


@dataclass
class WBFCluster:
    items: List[BoxItem]
    box: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 WBF-only validation baselines")

    p.add_argument("--data_yaml", type=str, default="/home/inspur/ishrai/yolo_server.yaml",
                   help="YOLO data yaml containing train/val/test image paths")
    p.add_argument("--split", type=str, default="valid", choices=["train", "valid", "val", "test"],
                   help="Dataset split to evaluate. Use valid/val for validation.")
    p.add_argument("--model_dir", type=str, default="/home/inspur/ishrai/all_togather",
                   help="Folder containing the four YOLO detector models")
    p.add_argument("--out_dir", type=str, default="/home/inspur/ishrai/phase1_wbf_valid",
                   help="Output folder")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--low_conf", type=float, default=0.001,
                   help="Low confidence threshold used when getting detector candidates")
    p.add_argument("--nms_iou", type=float, default=0.70,
                   help="Detector internal NMS IoU used before WBF candidate collection")
    p.add_argument("--max_det", type=int, default=300)

    p.add_argument("--wbf_iou", type=float, default=0.55,
                   help="WBF matching IoU threshold")
    p.add_argument("--save_conf", type=float, default=0.01,
                   help="Save fused candidates with confidence >= save_conf")
    p.add_argument("--eval_conf", type=float, default=0.25,
                   help="Fixed WBF-only operating threshold for validation metrics")
    p.add_argument("--match_iou", type=float, default=0.50,
                   help="IoU threshold for TP/FP/FN evaluation")

    p.add_argument("--limit", type=int, default=0,
                   help="Optional image limit for debugging. Use 0 for full split.")
    p.add_argument("--write_candidate_xlsx", action="store_true",
                   help="Also write all fused candidates to xlsx if row count is under Excel limit")

    return p.parse_args()


def cuda_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


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

    image_path_value = data[key]
    if isinstance(image_path_value, list):
        if len(image_path_value) != 1:
            raise ValueError("This script expects one image directory per split. Found a list with multiple entries.")
        image_path_value = image_path_value[0]

    img_dir = resolve_path(str(image_path_value), root, yaml_path.parent)
    if not img_dir.exists():
        raise FileNotFoundError(f"Image split path does not exist: {img_dir}")

    # Infer label directory by replacing an 'images' path component with 'labels'.
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
        raise FileNotFoundError(
            f"Label directory was inferred but not found: {label_dir}\n"
            f"Image directory: {img_dir}\n"
            "Please check your data yaml structure."
        )

    names_raw = data.get("names", DEFAULT_NAMES)
    if isinstance(names_raw, dict):
        names = {int(k): str(v) for k, v in names_raw.items()}
    else:
        names = {i: str(v) for i, v in enumerate(names_raw)}

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


def read_yolo_label(label_path: Path) -> List[Dict]:
    gts: List[Dict] = []
    if not label_path.exists():
        return gts
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
            gts.append({"class_id": cls, "box": np.array([x1, y1, x2, y2], dtype=float)})
    return gts


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=float)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-12
    return inter / union


def box_iou_pair(a: np.ndarray, b: np.ndarray) -> float:
    return float(box_iou_one_to_many(a, b.reshape(1, 4))[0])


def weighted_box(items: List[BoxItem], weights: Sequence[float]) -> np.ndarray:
    ws = np.array([max(1e-12, it.score) * weights[it.model_idx] for it in items], dtype=float)
    boxes = np.stack([it.box for it in items], axis=0)
    out = (boxes * ws[:, None]).sum(axis=0) / ws.sum()
    return np.clip(out, 0.0, 1.0)


def run_simple_wbf(
    preds_by_model: List[Dict[str, np.ndarray]],
    iou_thr: float,
    low_conf: float,
    save_conf: float,
    weights: Optional[Sequence[float]] = None,
) -> List[Dict]:
    """Simple class-wise WBF implementation for two or more detectors.

    Confidence type: avg.
    Confidence rescaling: min, implemented as mean(cluster_scores) * min(model_support, n_models) / n_models.
    """
    n_models = len(preds_by_model)
    if weights is None:
        weights = [1.0] * n_models

    items_by_class: Dict[int, List[BoxItem]] = {}
    for m_idx, pred in enumerate(preds_by_model):
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]
        for box, score, label in zip(boxes, scores, labels):
            score = float(score)
            label = int(label)
            if score < low_conf:
                continue
            items_by_class.setdefault(label, []).append(
                BoxItem(box=np.asarray(box, dtype=float), score=score, label=label, model_idx=m_idx)
            )

    fused: List[Dict] = []
    for label, items in items_by_class.items():
        items = sorted(items, key=lambda z: z.score, reverse=True)
        clusters: List[WBFCluster] = []

        for it in items:
            best_idx = -1
            best_iou = -1.0
            for c_idx, cl in enumerate(clusters):
                iou = box_iou_pair(it.box, cl.box)
                if iou >= iou_thr and iou > best_iou:
                    best_iou = iou
                    best_idx = c_idx
            if best_idx < 0:
                clusters.append(WBFCluster(items=[it], box=it.box.copy()))
            else:
                clusters[best_idx].items.append(it)
                clusters[best_idx].box = weighted_box(clusters[best_idx].items, weights)

        for cl in clusters:
            scores = np.array([it.score for it in cl.items], dtype=float)
            model_support = sorted(set(it.model_idx for it in cl.items))
            support_count = len(model_support)
            raw_mean = float(scores.mean()) if len(scores) else 0.0
            fused_score = raw_mean * min(support_count, n_models) / float(n_models)
            if fused_score >= save_conf:
                fused.append({
                    "box": cl.box.copy(),
                    "score": fused_score,
                    "score_raw_mean": raw_mean,
                    "class_id": int(label),
                    "support_count": int(support_count),
                    "num_boxes_in_cluster": int(len(cl.items)),
                    "model_support_indices": ";".join(str(x) for x in model_support),
                })

    fused.sort(key=lambda z: z["score"], reverse=True)
    return fused


def predict_yolo(model: YOLO, img_path: Path, args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], float, int]:
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
        return {
            "boxes": np.zeros((0, 4), dtype=float),
            "scores": np.zeros((0,), dtype=float),
            "labels": np.zeros((0,), dtype=int),
        }, elapsed, 0

    xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(float)
    scores = r.boxes.conf.detach().cpu().numpy().astype(float)
    labels = r.boxes.cls.detach().cpu().numpy().astype(int)

    xyxy[:, [0, 2]] /= float(w)
    xyxy[:, [1, 3]] /= float(h)
    xyxy = np.clip(xyxy, 0.0, 1.0)

    return {"boxes": xyxy, "scores": scores, "labels": labels}, elapsed, len(scores)


def evaluate_fixed_threshold(
    detections: List[Dict],
    gt_by_image: Dict[str, List[Dict]],
    class_names: Dict[int, str],
    conf_thr: float,
    iou_thr: float,
) -> Tuple[Dict, List[Dict]]:
    overall = {"TP": 0, "FP": 0, "FN": 0}
    per_class_rows: List[Dict] = []

    for cid, cname in class_names.items():
        tp = fp = fn = 0
        # group detections by image for this class
        dets_by_img: Dict[str, List[Dict]] = {}
        for d in detections:
            if int(d["class_id"]) == int(cid) and float(d["score"]) >= conf_thr:
                dets_by_img.setdefault(d["image_id"], []).append(d)

        for image_id, gt_list in gt_by_image.items():
            gt_cls = [g for g in gt_list if int(g["class_id"]) == int(cid)]
            matched = [False] * len(gt_cls)
            dets = sorted(dets_by_img.get(image_id, []), key=lambda z: z["score"], reverse=True)
            for d in dets:
                if gt_cls:
                    gt_boxes = np.stack([g["box"] for g in gt_cls], axis=0)
                    ious = box_iou_one_to_many(np.asarray(d["box"], dtype=float), gt_boxes)
                    best_idx = int(np.argmax(ious)) if len(ious) else -1
                    best_iou = float(ious[best_idx]) if best_idx >= 0 else 0.0
                else:
                    best_idx = -1
                    best_iou = 0.0
                if best_idx >= 0 and best_iou >= iou_thr and not matched[best_idx]:
                    tp += 1
                    matched[best_idx] = True
                else:
                    fp += 1
            fn += matched.count(False)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr_custom = fp / (tp + fp) if (tp + fp) else 0.0

        per_class_rows.append({
            "class_id": cid,
            "class_name": cname,
            "eval_conf": conf_thr,
            "match_iou": iou_thr,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "custom_fpr_fp_over_tp_fp": fpr_custom,
        })
        overall["TP"] += tp
        overall["FP"] += fp
        overall["FN"] += fn

    TP, FP, FN = overall["TP"], overall["FP"], overall["FN"]
    P = TP / (TP + FP) if (TP + FP) else 0.0
    R = TP / (TP + FN) if (TP + FN) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    FPR = FP / (TP + FP) if (TP + FP) else 0.0
    metrics = {
        "eval_conf": conf_thr,
        "match_iou": iou_thr,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": P,
        "recall": R,
        "f1": F1,
        "custom_fpr_fp_over_tp_fp": FPR,
    }
    return metrics, per_class_rows


def ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP."""
    if rec.size == 0:
        return 0.0
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        p = prec[rec >= t]
        ap += float(p.max()) if p.size else 0.0
    return ap / 101.0


def compute_ap_for_class(
    detections: List[Dict],
    gt_by_image: Dict[str, List[Dict]],
    class_id: int,
    iou_thr: float,
) -> float:
    gt_count = sum(1 for gts in gt_by_image.values() for g in gts if int(g["class_id"]) == int(class_id))
    if gt_count == 0:
        return 0.0

    dets = [d for d in detections if int(d["class_id"]) == int(class_id)]
    dets = sorted(dets, key=lambda z: float(z["score"]), reverse=True)
    if not dets:
        return 0.0

    matched_by_image: Dict[str, List[bool]] = {}
    gt_cls_by_image: Dict[str, List[Dict]] = {}
    for image_id, gts in gt_by_image.items():
        gt_cls = [g for g in gts if int(g["class_id"]) == int(class_id)]
        gt_cls_by_image[image_id] = gt_cls
        matched_by_image[image_id] = [False] * len(gt_cls)

    tp = np.zeros(len(dets), dtype=float)
    fp = np.zeros(len(dets), dtype=float)

    for i, d in enumerate(dets):
        image_id = d["image_id"]
        gt_cls = gt_cls_by_image.get(image_id, [])
        if gt_cls:
            gt_boxes = np.stack([g["box"] for g in gt_cls], axis=0)
            ious = box_iou_one_to_many(np.asarray(d["box"], dtype=float), gt_boxes)
            best_idx = int(np.argmax(ious)) if len(ious) else -1
            best_iou = float(ious[best_idx]) if best_idx >= 0 else 0.0
        else:
            best_idx = -1
            best_iou = 0.0

        if best_idx >= 0 and best_iou >= iou_thr and not matched_by_image[image_id][best_idx]:
            tp[i] = 1.0
            matched_by_image[image_id][best_idx] = True
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    rec = cum_tp / max(gt_count, 1)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    return ap_from_pr(rec, prec)


def compute_map_metrics(
    detections: List[Dict],
    gt_by_image: Dict[str, List[Dict]],
    class_names: Dict[int, str],
) -> Tuple[Dict, List[Dict]]:
    ap_rows: List[Dict] = []
    thresholds = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]

    for cid, cname in class_names.items():
        aps = []
        for thr in thresholds:
            ap = compute_ap_for_class(detections, gt_by_image, int(cid), float(thr))
            aps.append(ap)
            ap_rows.append({
                "class_id": int(cid),
                "class_name": cname,
                "iou_threshold": float(thr),
                "AP": ap,
            })
        ap50 = aps[0]
        ap5095 = float(np.mean(aps)) if aps else 0.0
        ap_rows.append({
            "class_id": int(cid),
            "class_name": cname,
            "iou_threshold": "0.50:0.95_mean",
            "AP": ap5095,
        })

    ap50_values = [r["AP"] for r in ap_rows if r["iou_threshold"] == 0.50]
    ap5095_values = [r["AP"] for r in ap_rows if r["iou_threshold"] == "0.50:0.95_mean"]
    out = {
        "mAP50_custom": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "mAP50_95_custom": float(np.mean(ap5095_values)) if ap5095_values else 0.0,
    }
    return out, ap_rows


def safe_write_excel(
    xlsx_path: Path,
    sheets: Dict[str, pd.DataFrame],
) -> None:
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            # Excel sheet names must be <=31 chars.
            safe_name = name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
            ws = writer.book[safe_name]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col[:2000]:  # cap for speed
                    try:
                        max_len = max(max_len, len(str(cell.value)) if cell.value is not None else 0)
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 38)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir).expanduser().resolve()
    model_paths = {
        "y8x": model_dir / "y8x_best.pt",
        "y8l": model_dir / "y8l_best.pt",
        "y26n": model_dir / "y26n_best.pt",
        "y26s": model_dir / "y26s_best.pt",
    }
    missing = [str(p) for p in model_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model file(s):\n" + "\n".join(missing))

    img_dir, label_dir, class_names = load_split_from_yaml(args.data_yaml, args.split)
    images = list_images(img_dir, limit=args.limit)

    print("=" * 80)
    print("Phase 1: WBF-only validation baselines")
    print(f"Data yaml      : {args.data_yaml}")
    print(f"Image dir      : {img_dir}")
    print(f"Label dir      : {label_dir}")
    print(f"Images         : {len(images)}")
    print(f"Class names    : {class_names}")
    print(f"Output dir     : {out_dir}")
    print("Models:")
    for k, v in model_paths.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    settings = {
        "data_yaml": str(Path(args.data_yaml).expanduser().resolve()),
        "split": args.split,
        "image_dir": str(img_dir),
        "label_dir": str(label_dir),
        "model_dir": str(model_dir),
        "models": {k: str(v) for k, v in model_paths.items()},
        "detector_low_conf": args.low_conf,
        "detector_nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "imgsz": args.imgsz,
        "wbf_iou": args.wbf_iou,
        "wbf_weights": "equal",
        "wbf_conf_type": "avg",
        "wbf_conf_rescale": "min",
        "save_conf": args.save_conf,
        "eval_conf": args.eval_conf,
        "match_iou": args.match_iou,
        "device": args.device,
        "num_images": len(images),
        "note": "Validation split is used for method/threshold selection only. Test split must not be used in Phase 1.",
    }
    (out_dir / "phase1_wbf_only_valid_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    # Ground truth cache
    gt_by_image: Dict[str, List[Dict]] = {}
    gt_rows: List[Dict] = []
    for img_path in images:
        image_id = image_id_from_path(img_path, img_dir)
        lab_path = label_path_for_image(img_path, img_dir, label_dir)
        gts = read_yolo_label(lab_path)
        gt_by_image[image_id] = gts
        for g in gts:
            b = g["box"]
            gt_rows.append({
                "image_id": image_id,
                "image_path": str(img_path),
                "label_path": str(lab_path),
                "class_id": int(g["class_id"]),
                "class_name": class_names.get(int(g["class_id"]), str(g["class_id"])),
                "x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
            })

    print("Loading YOLO models...")
    models = {name: YOLO(str(path)) for name, path in model_paths.items()}

    # Warm up models to reduce first-image timing distortion.
    first_img = images[0]
    print("Warming up models on one image...")
    for name, model in models.items():
        _pred, _elapsed, _n = predict_yolo(model, first_img, args)
        print(f"  warmup OK: {name}")

    pair_names = list(itertools.combinations(model_paths.keys(), 2))
    print("Detector pairs:")
    for a, b in pair_names:
        print(f"  {a}+{b}")

    all_det_rows: List[Dict] = []
    all_timing_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    per_class_metric_rows: List[Dict] = []
    ap_all_rows: List[Dict] = []

    for pair_idx, (m1_name, m2_name) in enumerate(pair_names, start=1):
        pair_name = f"{m1_name}+{m2_name}"
        print("\n" + "-" * 80)
        print(f"Running pair {pair_idx}/{len(pair_names)}: {pair_name}")
        pair_start = time.perf_counter()
        pair_det_rows: List[Dict] = []
        pair_timing_rows: List[Dict] = []

        m1 = models[m1_name]
        m2 = models[m2_name]

        for idx, img_path in enumerate(images, start=1):
            image_id = image_id_from_path(img_path, img_dir)

            pred1, t1, n1 = predict_yolo(m1, img_path, args)
            pred2, t2, n2 = predict_yolo(m2, img_path, args)

            t_wbf0 = time.perf_counter()
            fused = run_simple_wbf(
                [pred1, pred2],
                iou_thr=args.wbf_iou,
                low_conf=args.low_conf,
                save_conf=args.save_conf,
                weights=[1.0, 1.0],
            )
            t_wbf = time.perf_counter() - t_wbf0

            image_total = t1 + t2 + t_wbf
            num_eval = sum(1 for d in fused if float(d["score"]) >= args.eval_conf)

            pair_timing_rows.append({
                "pair": pair_name,
                "model_1": m1_name,
                "model_2": m2_name,
                "image_id": image_id,
                "image_path": str(img_path),
                "model_1_inference_sec": t1,
                "model_2_inference_sec": t2,
                "wbf_sec": t_wbf,
                "total_sec_model1_model2_wbf": image_total,
                "model_1_raw_predictions": n1,
                "model_2_raw_predictions": n2,
                "fused_candidates_saved": len(fused),
                "fused_candidates_at_eval_conf": num_eval,
            })

            for det_idx, d in enumerate(fused, start=1):
                b = d["box"]
                row = {
                    "pair": pair_name,
                    "model_1": m1_name,
                    "model_2": m2_name,
                    "image_id": image_id,
                    "image_path": str(img_path),
                    "det_id_in_image": det_idx,
                    "class_id": int(d["class_id"]),
                    "class_name": class_names.get(int(d["class_id"]), str(d["class_id"])),
                    "score": float(d["score"]),
                    "score_raw_mean_before_min_rescale": float(d["score_raw_mean"]),
                    "support_count": int(d["support_count"]),
                    "num_boxes_in_cluster": int(d["num_boxes_in_cluster"]),
                    "model_support_indices": d["model_support_indices"],
                    "x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
                    "saved_with_save_conf": args.save_conf,
                    "used_for_eval_at_conf_0p25": bool(float(d["score"]) >= args.eval_conf),
                }
                pair_det_rows.append(row)

            if idx % 100 == 0 or idx == len(images):
                elapsed_so_far = time.perf_counter() - pair_start
                ips = idx / elapsed_so_far if elapsed_so_far > 0 else 0.0
                print(f"  {idx}/{len(images)} images | {ips:.2f} images/s | saved dets so far: {len(pair_det_rows)}")

        pair_elapsed_wall = time.perf_counter() - pair_start
        pair_dets_for_eval = [dict(r, box=np.array([r["x1"], r["y1"], r["x2"], r["y2"]], dtype=float)) for r in pair_det_rows]
        fixed_metrics, pc_rows = evaluate_fixed_threshold(
            pair_dets_for_eval,
            gt_by_image,
            class_names,
            conf_thr=args.eval_conf,
            iou_thr=args.match_iou,
        )
        map_metrics, ap_rows = compute_map_metrics(pair_dets_for_eval, gt_by_image, class_names)

        total_model1 = sum(r["model_1_inference_sec"] for r in pair_timing_rows)
        total_model2 = sum(r["model_2_inference_sec"] for r in pair_timing_rows)
        total_wbf = sum(r["wbf_sec"] for r in pair_timing_rows)
        total_seq = sum(r["total_sec_model1_model2_wbf"] for r in pair_timing_rows)
        n_images = len(images)
        images_per_sec_wall = n_images / pair_elapsed_wall if pair_elapsed_wall > 0 else 0.0
        images_per_sec_seq = n_images / total_seq if total_seq > 0 else 0.0

        summary = {
            "pair": pair_name,
            "model_1": m1_name,
            "model_2": m2_name,
            "split": args.split,
            "num_images": n_images,
            "num_gt_boxes": len(gt_rows),
            "saved_candidates_conf_ge_save_conf": len(pair_det_rows),
            "eval_candidates_conf_ge_eval_conf": sum(1 for r in pair_det_rows if float(r["score"]) >= args.eval_conf),
            **fixed_metrics,
            **map_metrics,
            "wall_time_sec": pair_elapsed_wall,
            "sequential_model1_model2_wbf_sec": total_seq,
            "model_1_total_inference_sec": total_model1,
            "model_2_total_inference_sec": total_model2,
            "wbf_total_sec": total_wbf,
            "images_per_sec_wall": images_per_sec_wall,
            "images_per_sec_sequential_timed": images_per_sec_seq,
            "avg_ms_per_image_wall": 1000.0 * pair_elapsed_wall / n_images,
            "avg_ms_per_image_seq": 1000.0 * total_seq / n_images,
            "detector_low_conf": args.low_conf,
            "detector_nms_iou": args.nms_iou,
            "wbf_iou": args.wbf_iou,
            "save_conf": args.save_conf,
            "eval_conf": args.eval_conf,
            "match_iou": args.match_iou,
            "imgsz": args.imgsz,
            "device": args.device,
        }
        summary_rows.append(summary)

        for r in pc_rows:
            r["pair"] = pair_name
            r["model_1"] = m1_name
            r["model_2"] = m2_name
            per_class_metric_rows.append(r)

        for r in ap_rows:
            r["pair"] = pair_name
            r["model_1"] = m1_name
            r["model_2"] = m2_name
            ap_all_rows.append(r)

        all_det_rows.extend(pair_det_rows)
        all_timing_rows.extend(pair_timing_rows)

        print(f"Finished {pair_name}")
        print(f"  Precision : {fixed_metrics['precision']:.6f}")
        print(f"  Recall    : {fixed_metrics['recall']:.6f}")
        print(f"  F1        : {fixed_metrics['f1']:.6f}")
        print(f"  FP        : {fixed_metrics['FP']}")
        print(f"  FN        : {fixed_metrics['FN']}")
        print(f"  mAP50     : {map_metrics['mAP50_custom']:.6f}")
        print(f"  mAP50-95  : {map_metrics['mAP50_95_custom']:.6f}")
        print(f"  Throughput: {images_per_sec_wall:.3f} images/s wall, {images_per_sec_seq:.3f} images/s timed")

    # Save core outputs.
    df_summary = pd.DataFrame(summary_rows).sort_values(["f1", "precision", "recall"], ascending=False)
    df_per_class = pd.DataFrame(per_class_metric_rows)
    df_ap = pd.DataFrame(ap_all_rows)
    df_timing = pd.DataFrame(all_timing_rows)
    df_dets = pd.DataFrame(all_det_rows)
    df_gt = pd.DataFrame(gt_rows)
    df_settings = pd.DataFrame([settings])

    summary_xlsx = out_dir / "phase1_wbf_only_valid_summary.xlsx"
    safe_write_excel(summary_xlsx, {
        "summary_by_pair": df_summary,
        "per_class_metrics": df_per_class,
        "ap_by_class_iou": df_ap,
        "settings": df_settings,
        "gt_summary_sample": df_gt.head(5000),
    })

    det_csv = out_dir / "phase1_wbf_only_valid_all_fused_candidates.csv"
    timing_csv = out_dir / "phase1_wbf_only_valid_timing_by_image.csv"
    gt_csv = out_dir / "phase1_wbf_only_valid_ground_truth.csv"
    df_dets.to_csv(det_csv, index=False)
    df_timing.to_csv(timing_csv, index=False)
    df_gt.to_csv(gt_csv, index=False)

    timing_xlsx = out_dir / "phase1_wbf_only_valid_timing_by_image.xlsx"
    safe_write_excel(timing_xlsx, {"timing_by_image": df_timing})

    # Excel row limit is 1,048,576. Avoid silently truncating large candidate tables.
    if args.write_candidate_xlsx:
        if len(df_dets) <= 1_000_000:
            det_xlsx = out_dir / "phase1_wbf_only_valid_all_fused_candidates.xlsx"
            safe_write_excel(det_xlsx, {"all_fused_candidates": df_dets})
        else:
            print(f"Candidate xlsx skipped because rows={len(df_dets)} exceeds Excel row limit. CSV was saved.")

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print(f"Summary Excel       : {summary_xlsx}")
    print(f"Timing Excel        : {timing_xlsx}")
    print(f"All candidates CSV  : {det_csv}")
    print(f"Timing CSV          : {timing_csv}")
    print(f"Ground truth CSV    : {gt_csv}")
    print(f"Settings JSON       : {out_dir / 'phase1_wbf_only_valid_settings.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
