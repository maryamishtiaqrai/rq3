#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a CLIP/YOLOCLS crop-level verifier dataset from D5-clean.
Classes: knife, pistol, non_weapon.

Positive crops: ground-truth YOLO boxes, plus optional WBF-matched positive crops.
Non-weapon crops: hard background false positives mined from WBF outputs of all dual pairs
of the provided YOLO detectors. Class-confusion boxes around real weapons are NOT saved
as non_weapon; optionally they are saved as hard positives using the matched GT class.

Default WBF mining settings follow the published-code style:
low_conf=0.001, final_conf=0.25, wbf_iou=0.55, conf_type=avg,
conf_rescale=min, equal model weights.

Example:
python create_clip_dataset_from_wbf_pairs.py \
  --data_yaml /path/to/D5_clean/data.yaml \
  --models y8x:/path/y8x_best.pt y8l:/path/y8l_best.pt y26n:/path/y26n_best.pt y26s:/path/y26s_best.pt \
  --out_root /path/to/CLIP_D5_WBF_Crops \
  --splits train valid test \
  --device cuda:0 \
  --balance_splits --overwrite
"""

import os, json, yaml, shutil, random, argparse, hashlib, itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TARGET_CLASS_NAMES = {"knife", "pistol"}

# ----------------------------- WBF helpers -----------------------------

def prefilter_boxes(boxes, scores, labels, weights, thr):
    new_boxes = {}
    for t in range(len(boxes)):
        if len(boxes[t]) != len(scores[t]) or len(boxes[t]) != len(labels[t]):
            continue
        for j in range(len(boxes[t])):
            score = float(scores[t][j])
            if score < thr:
                continue
            label = int(labels[t][j])
            x1, y1, x2, y2 = map(float, boxes[t][j])
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            if (x2 - x1) * (y2 - y1) <= 0:
                continue
            rec = [float(label), score * float(weights[t]), float(weights[t]), float(t), x1, y1, x2, y2]
            new_boxes.setdefault(label, []).append(rec)
    for k in new_boxes:
        arr = np.array(new_boxes[k], dtype=np.float32)
        new_boxes[k] = arr[arr[:, 1].argsort()[::-1]]
    return new_boxes


def bb_iou_array(boxes_xyxy, new_box_xyxy):
    if boxes_xyxy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    xA = np.maximum(boxes_xyxy[:, 0], new_box_xyxy[0])
    yA = np.maximum(boxes_xyxy[:, 1], new_box_xyxy[1])
    xB = np.minimum(boxes_xyxy[:, 2], new_box_xyxy[2])
    yB = np.minimum(boxes_xyxy[:, 3], new_box_xyxy[3])
    inter = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)
    areaA = np.maximum(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0) * np.maximum(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0)
    areaB = max(new_box_xyxy[2] - new_box_xyxy[0], 0) * max(new_box_xyxy[3] - new_box_xyxy[1], 0)
    return inter / np.maximum(areaA + areaB - inter, 1e-10)


def find_matching_box_fast(weighted_boxes, new_box, match_iou):
    if weighted_boxes.shape[0] == 0:
        return -1, match_iou
    ious = bb_iou_array(weighted_boxes[:, 4:8], new_box[4:8])
    ious[weighted_boxes[:, 0] != new_box[0]] = -1.0
    best_idx = int(np.argmax(ious)); best_iou = float(ious[best_idx])
    if best_iou <= match_iou:
        return -1, match_iou
    return best_idx, best_iou


def get_weighted_box(cluster_boxes, conf_type="avg"):
    fused = np.zeros(8, dtype=np.float32)
    conf_sum, conf_list = 0.0, []
    for b in cluster_boxes:
        ci = float(b[1])
        fused[4:8] += ci * b[4:8]
        conf_sum += ci; conf_list.append(ci)
    fused[0] = cluster_boxes[0][0]
    fused[3] = float(len(cluster_boxes))
    fused[1] = float(np.max(conf_list)) if conf_type == "max" and conf_list else conf_sum / max(len(cluster_boxes), 1)
    if conf_sum > 0:
        fused[4:8] /= conf_sum
    fused[2] = float(np.sum([b[2] for b in cluster_boxes]))
    return fused


def rescale_confidence(c, t, n, mode="min"):
    if mode == "none" or n <= 0: return c
    if mode == "linear": return c * (t / float(n))
    return c * (min(t, float(n)) / float(n))


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type="avg", conf_rescale="min"):
    num_models = len(boxes_list)
    weights = np.ones(num_models, dtype=np.float32) if weights is None else np.array(weights, dtype=np.float32)
    if len(weights) != num_models:
        weights = np.ones(num_models, dtype=np.float32)
    filtered = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if not filtered:
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    overall = []
    for label in filtered:
        boxes = filtered[label]
        clusters, fused_list = [], np.empty((0,8), dtype=np.float32)
        for j in range(len(boxes)):
            idx, _ = find_matching_box_fast(fused_list, boxes[j], iou_thr)
            if idx != -1:
                clusters[idx].append(boxes[j]); fused_list[idx] = get_weighted_box(clusters[idx], conf_type)
            else:
                clusters.append([boxes[j].copy()]); fused_list = np.vstack((fused_list, get_weighted_box(clusters[-1], conf_type)))
        for r in range(fused_list.shape[0]):
            fused_list[r, 1] = rescale_confidence(fused_list[r, 1], float(fused_list[r, 3]), num_models, conf_rescale)
        overall.append(fused_list)
    overall = np.concatenate(overall, axis=0)
    overall = overall[overall[:, 1].argsort()[::-1]]
    return overall[:, 4:8], overall[:, 1], overall[:, 0]

# ----------------------------- IO helpers -----------------------------

def parse_args():
    ap = argparse.ArgumentParser("Create CLIP verifier dataset using all dual WBF YOLO pairs")
    ap.add_argument("--data_yaml", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name:/path/model.pt, e.g. y8x:/a.pt y8l:/b.pt y26n:/c.pt y26s:/d.pt")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"], choices=["train", "valid", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--low_conf", type=float, default=0.001)
    ap.add_argument("--final_conf", type=float, default=0.25)
    ap.add_argument("--wbf_iou", type=float, default=0.55)
    ap.add_argument("--conf_type", default="avg", choices=["avg", "max"])
    ap.add_argument("--conf_rescale", default="min", choices=["min", "linear", "none"])
    ap.add_argument("--hard_iou_thr", type=float, default=0.10)
    ap.add_argument("--pred_overlap_thr", type=float, default=0.10)
    ap.add_argument("--negative_pad", type=float, default=0.05)
    ap.add_argument("--min_crop_size", type=int, default=16)
    ap.add_argument("--max_neg_per_image_pair", type=int, default=8)
    ap.add_argument("--include_wbf_matched_positives", action="store_true",
                    help="Also save WBF boxes that overlap a GT weapon as positive crops using the matched GT class. This includes class-confusion boxes such as a knife detected as pistol.")
    ap.add_argument("--positive_match_iou", type=float, default=0.50,
                    help="IoU threshold to save WBF-matched positive crops. Used only with --include_wbf_matched_positives.")
    ap.add_argument("--wbf_positive_pad", type=float, default=0.05,
                    help="Padding for optional WBF-matched positive crops.")
    ap.add_argument("--max_wbf_pos_per_image_pair", type=int, default=4,
                    help="Maximum optional WBF-matched positive crops per image per pair.")
    ap.add_argument("--train_pos_pads", default="0.00,0.15,0.30")
    ap.add_argument("--eval_pos_pads", default="0.15")
    ap.add_argument("--balance_splits", action="store_true")
    ap.add_argument("--balance_train_only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_images_per_split", type=int, default=None, help="debug only")
    return ap.parse_args()


def safe_stem(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(root: str, p: Any, yaml_dir: str) -> str:
    if isinstance(p, list):
        if len(p) != 1: raise ValueError("This script expects one image directory per split")
        p = p[0]
    p = str(p)
    if os.path.isabs(p): return os.path.normpath(p)
    base = yaml_dir if not root else (root if os.path.isabs(root) else os.path.join(yaml_dir, root))
    return os.path.normpath(os.path.join(base, p))


def get_split_image_dir(data: Dict[str, Any], yaml_path: str, split: str) -> str:
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    root = data.get("path", "")
    key = "val" if split in ("valid", "val") and "val" in data else ("valid" if split in ("valid", "val") else split)
    if key not in data: raise KeyError(f"Missing split key '{key}' in data.yaml")
    return resolve_path(root, data[key], yaml_dir)


def infer_labels_dir(images_dir: str) -> str:
    p = Path(images_dir); parts = list(p.parts)
    for idx in range(len(parts)-1, -1, -1):
        if parts[idx].lower() == "images":
            parts[idx] = "labels"; return str(Path(*parts))
    return str(p.parent.parent / "labels" / p.name)


def parse_names(data: Dict[str, Any]) -> Dict[int, str]:
    names = data.get("names")
    if names is None: return {0: "knife", 1: "pistol"}
    if isinstance(names, dict): return {int(k): str(v).strip().lower().replace(" ", "_") for k, v in names.items()}
    if isinstance(names, list): return {i: str(v).strip().lower().replace(" ", "_") for i, v in enumerate(names)}
    return {0: "knife", 1: "pistol"}


def list_images(images_dir: str) -> List[str]:
    files = [str(p) for p in Path(images_dir).rglob("*") if p.suffix.lower() in IMG_EXTS]
    files.sort(); return files


def label_path_for_image(img_path: str, images_dir: str, labels_dir: str) -> str:
    rel = Path(img_path).relative_to(images_dir)
    return str(Path(labels_dir) / rel.with_suffix(".txt"))


def read_yolo_labels(label_path: str) -> Tuple[List[List[float]], List[int]]:
    boxes, labels = [], []
    if not os.path.exists(label_path): return boxes, labels
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls = int(float(parts[0])); xc, yc, bw, bh = map(float, parts[1:5])
            x1, y1 = max(0, xc-bw/2), max(0, yc-bh/2)
            x2, y2 = min(1, xc+bw/2), min(1, yc+bh/2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1,y1,x2,y2]); labels.append(cls)
    return boxes, labels

# ----------------------------- box/crop helpers -----------------------------

def norm_to_px(box, w, h):
    x1,y1,x2,y2 = box
    return [int(round(max(0,min(1,x1))*w)), int(round(max(0,min(1,y1))*h)), int(round(max(0,min(1,x2))*w)), int(round(max(0,min(1,y2))*h))]


def add_pad(box, pad):
    x1,y1,x2,y2 = map(float, box); bw, bh = x2-x1, y2-y1
    return [max(0,x1-bw*pad), max(0,y1-bh*pad), min(1,x2+bw*pad), min(1,y2+bh*pad)]


def crop_from_box(img, box_norm, pad, min_size):
    h,w = img.shape[:2]
    x1,y1,x2,y2 = norm_to_px(add_pad(box_norm, pad), w, h)
    x1,y1,x2,y2 = max(0,min(w-1,x1)), max(0,min(h-1,y1)), max(0,min(w,x2)), max(0,min(h,y2))
    if x2 <= x1: x2 = min(w, x1+1)
    if y2 <= y1: y2 = min(h, y1+1)
    if (x2-x1) < min_size or (y2-y1) < min_size: return None
    crop = img[y1:y2, x1:x2].copy()
    return None if crop.size == 0 else (crop, [x1,y1,x2,y2])


def iou(a,b):
    x1,y1,x2,y2 = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    aa = max(a[2]-a[0],0)*max(a[3]-a[1],0); ab = max(b[2]-b[0],0)*max(b[3]-b[1],0)
    return inter / max(aa+ab-inter, 1e-10)


def inter_over_pred(pred, gt):
    x1,y1,x2,y2 = max(pred[0],gt[0]), max(pred[1],gt[1]), min(pred[2],gt[2]), min(pred[3],gt[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2-x1)*(y2-y1); area = max(pred[2]-pred[0],0)*max(pred[3]-pred[1],0)
    return inter / max(area, 1e-10)


def center_inside(pred, gts):
    cx, cy = (pred[0]+pred[2])/2, (pred[1]+pred[3])/2
    return any(g[0] <= cx <= g[2] and g[1] <= cy <= g[3] for g in gts)


def best_gt_match(pred, gt_boxes, gt_labels):
    """Return best IoU match against any GT box, regardless of class."""
    if not gt_boxes:
        return -1, 0.0, None
    vals = [iou(pred, g) for g in gt_boxes]
    idx = int(np.argmax(vals))
    return idx, float(vals[idx]), int(gt_labels[idx])


def hard_neg_ok(pred, gt_boxes, hard_iou_thr, pred_overlap_thr):
    # This test is against ALL ground-truth boxes, not only boxes of the predicted class.
    # Therefore, a knife detected as pistol, or a pistol detected as knife, will have
    # high overlap with a real weapon and will NOT become a non_weapon crop.
    max_iou = max([iou(pred,g) for g in gt_boxes], default=0.0)
    max_iop = max([inter_over_pred(pred,g) for g in gt_boxes], default=0.0)
    inside = center_inside(pred, gt_boxes)
    ok = max_iou <= hard_iou_thr and max_iop <= pred_overlap_thr and not inside
    return ok, max_iou, max_iop, int(inside)


def crop_hash(crop):
    small = cv2.resize(crop, (32,32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return hashlib.sha1(gray.tobytes()).hexdigest()


def write_crop(crop, path, quality):
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]))

# ----------------------------- model helpers -----------------------------

def parse_model_specs(specs):
    out = {}
    for spec in specs:
        if ":" not in spec: raise ValueError(f"Use name:/path/model.pt format: {spec}")
        name, path = spec.split(":", 1); name = safe_stem(name.strip()); path = path.strip()
        if not os.path.exists(path): raise FileNotFoundError(path)
        out[name] = path
    if len(out) < 2: raise ValueError("Need at least two models")
    return out


def predict_model(model, img_path, w, h, imgsz, conf, device):
    pred = model.predict(img_path, imgsz=imgsz, conf=conf, device=device, verbose=False)
    boxes, scores, labels = [], [], []
    for r in pred:
        for b in r.boxes:
            x1,y1,x2,y2 = b.xyxy.cpu().numpy()[0]
            boxes.append([float(x1/w), float(y1/h), float(x2/w), float(y2/h)])
            scores.append(float(b.conf.item())); labels.append(int(b.cls.item()))
    return boxes, scores, labels

# ----------------------------- generation -----------------------------

def save_positive(img, img_path, label_path, split, gt_boxes, gt_labels, class_map, pads, out_root, rows, seen, args):
    base = safe_stem(Path(img_path).stem)
    for gi, (box, lab) in enumerate(zip(gt_boxes, gt_labels)):
        cls = class_map.get(int(lab), f"class_{lab}")
        if cls not in TARGET_CLASS_NAMES: continue
        for pad in pads:
            res = crop_from_box(img, box, pad, args.min_crop_size)
            if res is None: continue
            crop, px = res; hsh = crop_hash(crop); key = (split, cls, hsh)
            if key in seen: continue
            seen.add(key)
            fname = f"{base}_gt{gi:04d}_pad{str(pad).replace('.', 'p')}.jpg"
            outp = out_root/"raw"/split/cls/fname
            if not write_crop(crop, outp, args.jpeg_quality): continue
            rows.append(dict(crop_path=str(outp), split=split, class_name=cls, class_id=0 if cls=="knife" else 1,
                             source_image=str(img_path), source_label=str(label_path), source_type="gt_positive",
                             detector_source="none", wbf_pair="none", detector_a="none", detector_b="none",
                             box_x1=px[0], box_y1=px[1], box_x2=px[2], box_y2=px[3],
                             box_x1_norm=box[0], box_y1_norm=box[1], box_x2_norm=box[2], box_y2_norm=box[3],
                             crop_padding=pad, fused_confidence="", predicted_label="", predicted_class_name="",
                             max_iou_with_gt="", max_intersection_over_pred_box="", center_inside_gt="",
                             matched_gt_label="", matched_gt_class_name="",
                             is_gt_crop=1, is_hard_negative=0, crop_hash=hsh))


def save_wbf_crops(img, img_path, label_path, split, gt_boxes, gt_labels, pair_name, a, b,
                  fused_boxes, fused_scores, fused_labels, class_map, out_root, rows, seen, args):
    """
    Save WBF-mined crops with safe label logic:
      1) Pure background false positives -> non_weapon.
      2) Boxes matching any GT weapon with IoU >= positive_match_iou -> optional positive crop,
         labeled by the matched GT class, not by the predicted class.
      3) Ambiguous partial-overlap boxes -> discarded.

    This prevents class-confusion boxes, e.g. a real knife predicted as pistol,
    from being incorrectly added to non_weapon.
    """
    base = safe_stem(Path(img_path).stem)
    saved_neg = 0
    saved_pos = 0

    for di, (box, score, lab) in enumerate(zip(fused_boxes, fused_scores, fused_labels)):
        box = list(map(float, box))
        pred_cls = class_map.get(int(lab), f"class_{int(lab)}")
        best_idx, best_iou_val, best_gt_label = best_gt_match(box, gt_boxes, gt_labels)
        max_iop = max([inter_over_pred(box, g) for g in gt_boxes], default=0.0)
        inside = int(center_inside(box, gt_boxes))

        # Case A: box matches a real weapon. Save optionally as a positive crop using TRUE GT class.
        if args.include_wbf_matched_positives and best_idx >= 0 and best_iou_val >= args.positive_match_iou:
            if saved_pos >= args.max_wbf_pos_per_image_pair:
                continue
            true_cls = class_map.get(int(best_gt_label), f"class_{int(best_gt_label)}")
            if true_cls not in TARGET_CLASS_NAMES:
                continue
            res = crop_from_box(img, box, args.wbf_positive_pad, args.min_crop_size)
            if res is None:
                continue
            crop, px = res
            hsh = crop_hash(crop)
            key = (split, true_cls, hsh)
            if key in seen:
                continue
            seen.add(key)
            source_type = "wbf_true_positive" if pred_cls == true_cls else "wbf_class_confusion_positive"
            fname = f"{base}_{pair_name}_{source_type}_det{pred_cls}_gt{true_cls}_{di:04d}_conf{float(score):.4f}.jpg"
            outp = out_root/"raw"/split/true_cls/fname
            if not write_crop(crop, outp, args.jpeg_quality):
                continue
            rows.append(dict(crop_path=str(outp), split=split, class_name=true_cls, class_id=0 if true_cls=="knife" else 1,
                             source_image=str(img_path), source_label=str(label_path), source_type=source_type,
                             detector_source="wbf", wbf_pair=pair_name, detector_a=a, detector_b=b,
                             box_x1=px[0], box_y1=px[1], box_x2=px[2], box_y2=px[3],
                             box_x1_norm=box[0], box_y1_norm=box[1], box_x2_norm=box[2], box_y2_norm=box[3],
                             crop_padding=args.wbf_positive_pad, fused_confidence=float(score), predicted_label=int(lab), predicted_class_name=pred_cls,
                             max_iou_with_gt=best_iou_val, max_intersection_over_pred_box=max_iop, center_inside_gt=inside,
                             matched_gt_label=int(best_gt_label), matched_gt_class_name=true_cls,
                             is_gt_crop=0, is_hard_negative=0, crop_hash=hsh))
            saved_pos += 1
            continue

        # Case B: pure background false positive -> non_weapon.
        if saved_neg < args.max_neg_per_image_pair:
            ok, miou, miop, inside_flag = hard_neg_ok(box, gt_boxes, args.hard_iou_thr, args.pred_overlap_thr)
            if ok:
                res = crop_from_box(img, box, args.negative_pad, args.min_crop_size)
                if res is None:
                    continue
                crop, px = res
                hsh = crop_hash(crop)
                key = (split, "non_weapon", hsh)
                if key in seen:
                    continue
                seen.add(key)
                fname = f"{base}_{pair_name}_fp{di:04d}_conf{float(score):.4f}.jpg"
                outp = out_root/"raw"/split/"non_weapon"/fname
                if not write_crop(crop, outp, args.jpeg_quality):
                    continue
                rows.append(dict(crop_path=str(outp), split=split, class_name="non_weapon", class_id=2,
                                 source_image=str(img_path), source_label=str(label_path), source_type="wbf_background_false_positive",
                                 detector_source="wbf", wbf_pair=pair_name, detector_a=a, detector_b=b,
                                 box_x1=px[0], box_y1=px[1], box_x2=px[2], box_y2=px[3],
                                 box_x1_norm=box[0], box_y1_norm=box[1], box_x2_norm=box[2], box_y2_norm=box[3],
                                 crop_padding=args.negative_pad, fused_confidence=float(score), predicted_label=int(lab), predicted_class_name=pred_cls,
                                 max_iou_with_gt=miou, max_intersection_over_pred_box=miop, center_inside_gt=inside_flag,
                                 matched_gt_label="", matched_gt_class_name="",
                                 is_gt_crop=0, is_hard_negative=1, crop_hash=hsh))
                saved_neg += 1
                continue

        # Case C: ambiguous partial overlap or class confusion not saved as positive -> discard.
        # We do not write discarded crops because they can introduce label noise.

def process_split(split, data, data_yaml, models, pairs, class_map, out_root, args):
    split_name = "valid" if split == "val" else split
    img_dir = get_split_image_dir(data, data_yaml, split)
    lab_dir = infer_labels_dir(img_dir)
    if not os.path.isdir(img_dir): raise FileNotFoundError(img_dir)
    if not os.path.isdir(lab_dir): raise FileNotFoundError(lab_dir)
    imgs = list_images(img_dir)
    if args.max_images_per_split: imgs = imgs[:args.max_images_per_split]
    pads = [float(x) for x in (args.train_pos_pads if split_name=="train" else args.eval_pos_pads).split(",") if x.strip()]
    print(f"\nSplit={split_name} | images={len(imgs)} | image_dir={img_dir} | label_dir={lab_dir}")
    rows, seen = [], set()
    for img_path in tqdm(imgs, desc=f"{split_name}"):
        img = cv2.imread(img_path)
        if img is None: continue
        h,w = img.shape[:2]
        label_path = label_path_for_image(img_path, img_dir, lab_dir)
        gt_boxes, gt_labels = read_yolo_labels(label_path)
        save_positive(img, img_path, label_path, split_name, gt_boxes, gt_labels, class_map, pads, out_root, rows, seen, args)
        cache = {name: predict_model(m, img_path, w, h, args.imgsz, args.low_conf, args.device) for name, m in models.items()}
        for a,b in pairs:
            boxes_a, scores_a, labels_a = cache[a]; boxes_b, scores_b, labels_b = cache[b]
            fboxes, fscores, flabels = weighted_boxes_fusion([boxes_a, boxes_b], [scores_a, scores_b], [labels_a, labels_b],
                                                            weights=None, iou_thr=args.wbf_iou, skip_box_thr=args.low_conf,
                                                            conf_type=args.conf_type, conf_rescale=args.conf_rescale)
            if len(fscores) > 0:
                keep = fscores >= args.final_conf; fboxes, fscores, flabels = fboxes[keep], fscores[keep], flabels[keep]
            save_wbf_crops(img, img_path, label_path, split_name, gt_boxes, gt_labels, f"{a}+{b}", a, b, fboxes, fscores, flabels, class_map, out_root, rows, seen, args)
    df = pd.DataFrame(rows)
    df.to_csv(out_root/f"{split_name}_metadata_raw.csv", index=False)
    print(f"Raw crop counts for {split_name}:\n{df['class_name'].value_counts().to_string() if not df.empty else 'EMPTY'}")
    return df


def copy_balanced(df, out_root, split, seed):
    req = ["knife", "pistol", "non_weapon"]
    if df.empty or any(c not in set(df["class_name"]) for c in req):
        print(f"Skipping balanced copy for {split}: missing one or more classes")
        return pd.DataFrame()
    n = min(int((df["class_name"]==c).sum()) for c in req)
    rng = np.random.default_rng(seed)
    parts = []
    for c in req:
        cdf = df[df["class_name"]==c]
        parts.append(cdf.loc[rng.choice(cdf.index.to_numpy(), size=n, replace=False)])
    outdf = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    new_paths = []
    for i, row in tqdm(outdf.iterrows(), total=len(outdf), desc=f"balance {split}", leave=False):
        src = Path(row["crop_path"]); dst = out_root/"balanced"/split/row["class_name"]/f"{i:07d}_{src.name}"
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); new_paths.append(str(dst))
    outdf["balanced_crop_path"] = new_paths
    outdf.to_csv(out_root/f"{split}_metadata_balanced.csv", index=False)
    print(f"Balanced crop counts for {split}:\n{outdf['class_name'].value_counts().to_string()}")
    return outdf


def main():
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed)
    out_root = Path(args.out_root).resolve()
    if out_root.exists() and args.overwrite: shutil.rmtree(out_root)
    for split in ["valid" if s=="val" else s for s in args.splits]:
        for cls in ["knife", "pistol", "non_weapon"]: (out_root/"raw"/split/cls).mkdir(parents=True, exist_ok=True)
    data_yaml = os.path.abspath(args.data_yaml); data = load_yaml(data_yaml); class_map = parse_names(data)
    print("Class map:", class_map)
    model_paths = parse_model_specs(args.models)
    models = {name: YOLO(path) for name, path in model_paths.items()}
    pairs = list(itertools.combinations(list(model_paths.keys()), 2))
    print("Models:", list(model_paths.keys()))
    print("All dual pairs:", [f"{a}+{b}" for a,b in pairs])
    raw_dfs, bal_dfs = {}, {}
    for split in ["valid" if s=="val" else s for s in args.splits]:
        df = process_split(split, data, data_yaml, models, pairs, class_map, out_root, args)
        raw_dfs[split] = df
        if args.balance_splits and not (args.balance_train_only and split != "train"):
            bal_dfs[split] = copy_balanced(df, out_root, split, args.seed)
    summary = {
        "purpose": "CLIP/YOLOCLS verifier dataset from WBF hard negatives",
        "wbf_mining_settings": dict(low_conf=args.low_conf, final_conf=args.final_conf, wbf_iou=args.wbf_iou, conf_type=args.conf_type, conf_rescale=args.conf_rescale, weights="equal"),
        "labeling_rules": dict(non_weapon="max_iou_with_all_gt<=hard_iou_thr AND max_intersection_over_pred_box<=pred_overlap_thr AND center_outside_all_gt", class_confusion="never saved as non_weapon; optionally saved as positive crop with matched GT class", matched_positive_iou=args.positive_match_iou, include_wbf_matched_positives=args.include_wbf_matched_positives),
        "hard_negative_rules": dict(max_iou_with_gt=args.hard_iou_thr, max_intersection_over_pred_box=args.pred_overlap_thr, center_inside_gt=False, negative_pad=args.negative_pad, max_neg_per_image_pair=args.max_neg_per_image_pair),
        "models": model_paths,
        "pairs": [f"{a}+{b}" for a,b in pairs],
        "raw_counts": {s: (df["class_name"].value_counts().to_dict() if not df.empty else {}) for s,df in raw_dfs.items()},
        "balanced_counts": {s: (df["class_name"].value_counts().to_dict() if df is not None and not df.empty else {}) for s,df in bal_dfs.items()},
        "split_safety": "train crops from train only; valid crops from valid only; test crops from test only"
    }
    with open(out_root/"dataset_creation_summary.json", "w", encoding="utf-8") as f: json.dump(summary, f, indent=2)
    with pd.ExcelWriter(out_root/"dataset_creation_summary.xlsx") as writer:
        pd.DataFrame([summary["wbf_mining_settings"]]).to_excel(writer, sheet_name="WBF_Settings", index=False)
        pd.DataFrame([summary["hard_negative_rules"]]).to_excel(writer, sheet_name="HardNegativeRules", index=False)
        pd.DataFrame([summary["labeling_rules"]]).to_excel(writer, sheet_name="LabelingRules", index=False)
        pd.DataFrame({"model": list(model_paths.keys()), "path": list(model_paths.values())}).to_excel(writer, sheet_name="Models", index=False)
        pd.DataFrame({"pair": summary["pairs"]}).to_excel(writer, sheet_name="Pairs", index=False)
        for s, df in raw_dfs.items():
            (df["class_name"].value_counts().rename_axis("class_name").reset_index(name="count") if not df.empty else pd.DataFrame(columns=["class_name","count"])).to_excel(writer, sheet_name=f"{s}_raw", index=False)
        for s, df in bal_dfs.items():
            (df["class_name"].value_counts().rename_axis("class_name").reset_index(name="count") if df is not None and not df.empty else pd.DataFrame(columns=["class_name","count"])).to_excel(writer, sheet_name=f"{s}_balanced", index=False)
    print("\nDone")
    print("Raw dataset:", out_root/"raw")
    if args.balance_splits: print("Balanced dataset:", out_root/"balanced")
    print("Summary:", out_root/"dataset_creation_summary.xlsx")

if __name__ == "__main__":
    main()
