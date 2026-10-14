#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-only CC-WBF component ablation + CLIP filter-recovery evaluation.

Purpose
-------
This script tests all 8 combinations of the three CC-WBF components:

T = transitive clustering
W = self-weighted confidence aggregation
U = unique-contributor confidence rescaling

000 = standard WBF-like setting
100 = T only
010 = W only
001 = U only
110 = T + W
101 = T + U
011 = W + U
111 = full CC-WBF

Protocol
--------
1. Run all 8 variants on the main validation split.
2. Select the best variant and threshold using validation F1 only.
3. Freeze the selected setting.
4. Evaluate on the main test split.
5. Optionally apply the same frozen setting to an external zero-shot dataset.

No model retraining is performed. No threshold or variant selection is done on the
main test split or external dataset.

Classes
-------
0 = knife
1 = pistol / handgun treated as pistol
"""

import csv
import json
import math
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import open_clip


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_IDS = [0, 1]
CLASS_NAMES = {0: "knife", 1: "pistol"}

CLIP_PROMPTS = [
    "a cropped image of a knife",
    "a cropped image of a pistol",
    "a cropped image of a non-weapon object",
]

VARIANTS = ["000", "100", "010", "001", "110", "101", "011", "111"]
VARIANT_NAMES = {
    "000": "WBF-like",
    "100": "T only",
    "010": "W only",
    "001": "U only",
    "110": "T+W",
    "101": "T+U",
    "011": "W+U",
    "111": "T+W+U full CC-WBF",
}


def variant_components(variant: str) -> Dict[str, bool]:
    if len(variant) != 3 or any(ch not in "01" for ch in variant):
        raise ValueError(f"Invalid variant '{variant}'. Use a 3-bit string such as 000 or 111.")
    return {
        "T_transitive_clustering": variant[0] == "1",
        "W_self_weighted_confidence": variant[1] == "1",
        "U_unique_contributor_rescaling": variant[2] == "1",
    }


def parse_float_list(text: str) -> List[float]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError(f"Could not parse float list from: {text}")
    return vals


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def round_value(x):
    if isinstance(x, float):
        return round(x, 4)
    return x


def round_dict(d: Dict) -> Dict:
    return {k: round_value(v) for k, v in d.items()}


def box_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def xywhn_to_xyxyn(xywhn: List[float]) -> List[float]:
    x, y, w, h = xywhn
    x1 = max(0.0, x - w / 2.0)
    y1 = max(0.0, y - h / 2.0)
    x2 = min(1.0, x + w / 2.0)
    y2 = min(1.0, y + h / 2.0)
    return [x1, y1, x2, y2]


def norm_to_pixel_box(box: List[float], img_w: int, img_h: int) -> List[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(img_w - 1, int(round(x1 * img_w))))
    y1 = max(0, min(img_h - 1, int(round(y1 * img_h))))
    x2 = max(0, min(img_w, int(round(x2 * img_w))))
    y2 = max(0, min(img_h, int(round(y2 * img_h))))

    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    return [x1, y1, x2, y2]


def pad_pixel_box(box: List[int], img_w: int, img_h: int, pad_ratio: float = 0.05) -> List[int]:
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * pad_ratio))
    py = int(round(bh * pad_ratio))

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(img_w, x2 + px)
    y2 = min(img_h, y2 + py)

    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    return [x1, y1, x2, y2]


def find_images(img_dir: Path) -> List[Path]:
    return sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def label_path_from_image(image_path: Path, image_root: Path, label_root: Path) -> Path:
    rel = image_path.relative_to(image_root)
    return label_root / rel.with_suffix(".txt")


def load_yolo_gt(label_path: Path) -> List[Dict]:
    gts = []
    if not label_path.is_file():
        return gts

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            if cls not in CLASS_IDS:
                continue

            xywhn = list(map(float, parts[1:5]))
            box = xywhn_to_xyxyn(xywhn)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            gts.append({"box": box, "cls": cls})

    return gts


def load_split_from_root(dataset_root: Path, split: str) -> Tuple[List[Path], Path, Path]:
    image_root = dataset_root / "images" / split
    label_root = dataset_root / "labels" / split

    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image split folder: {image_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"Missing label split folder: {label_root}")

    images = find_images(image_root)
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in: {image_root}")

    return images, image_root, label_root


def load_dataset_from_images_labels(image_root: Path, label_root: Path) -> Tuple[List[Path], Path, Path]:
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image folder: {image_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"Missing label folder: {label_root}")

    images = find_images(image_root)
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in: {image_root}")

    return images, image_root, label_root


@torch.no_grad()
def yolo_predict_normxyxy(
    model: YOLO,
    image_path: Path,
    img_w: int,
    img_h: int,
    imgsz: int,
    conf_th: float,
    iou_th: float,
    max_det: int,
    device: str,
) -> List[Dict]:
    res = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf_th,
        iou=iou_th,
        max_det=max_det,
        device=0 if str(device).startswith("cuda") else device,
        verbose=False,
    )

    if not res:
        return []

    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    boxes = r0.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = r0.boxes.conf.detach().cpu().numpy().astype(np.float32)
    labels = r0.boxes.cls.detach().cpu().numpy().astype(np.int64)

    boxes[:, [0, 2]] /= float(img_w)
    boxes[:, [1, 3]] /= float(img_h)
    boxes = np.clip(boxes, 0.0, 1.0)

    preds = []
    for b, s, c in zip(boxes, scores, labels):
        cls = int(c)
        if cls not in CLASS_IDS:
            continue
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        preds.append({"box": b.astype(float).tolist(), "conf": float(s), "cls": cls})

    return preds


# -------------------------------------------------------------------------
# Fusion variants
# -------------------------------------------------------------------------
def _iou_matrix_xyxy(boxes_xyxy: np.ndarray) -> np.ndarray:
    n = boxes_xyxy.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    x1 = boxes_xyxy[:, 0][:, None]
    y1 = boxes_xyxy[:, 1][:, None]
    x2 = boxes_xyxy[:, 2][:, None]
    y2 = boxes_xyxy[:, 3][:, None]

    xx1 = np.maximum(x1, x1.T)
    yy1 = np.maximum(y1, y1.T)
    xx2 = np.minimum(x2, x2.T)
    yy2 = np.minimum(y2, y2.T)

    inter_w = np.maximum(xx2 - xx1, 0.0)
    inter_h = np.maximum(yy2 - yy1, 0.0)
    inter = inter_w * inter_h

    area = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    union = area + area.T - inter
    return inter / np.maximum(union, 1e-10)


def _connected_components(adj: np.ndarray) -> List[List[int]]:
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    comps = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = [i]

        while stack:
            u = stack.pop()
            for v in np.where(adj[u])[0]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    comp.append(v)
        comps.append(comp)

    return comps


def _fused_box_for_indices(arr: np.ndarray, indices: List[int]) -> List[float]:
    cluster = arr[np.array(indices, dtype=np.int32)]
    conf_arr = cluster[:, 1].astype(np.float32)
    s1 = float(np.sum(conf_arr))
    if s1 <= 0.0:
        fused_box = np.mean(cluster[:, 3:7], axis=0).astype(np.float32)
    else:
        fused_box = (np.sum(cluster[:, 3:7] * conf_arr[:, None], axis=0) / s1).astype(np.float32)
    return fused_box.astype(float).tolist()


def _direct_clusters(arr: np.ndarray, iou_thr: float) -> List[List[int]]:
    """Standard WBF-like direct-overlap grouping."""
    clusters: List[List[int]] = []
    fused_boxes: List[List[float]] = []

    for i in range(arr.shape[0]):
        box = arr[i, 3:7].astype(float).tolist()
        best_iou = 0.0
        best_idx = -1

        for cidx, fbox in enumerate(fused_boxes):
            val = box_iou(box, fbox)
            if val > best_iou:
                best_iou = val
                best_idx = cidx

        if best_idx >= 0 and best_iou > float(iou_thr):
            clusters[best_idx].append(i)
            fused_boxes[best_idx] = _fused_box_for_indices(arr, clusters[best_idx])
        else:
            clusters.append([i])
            fused_boxes.append(box)

    return clusters


def _transitive_clusters(arr: np.ndarray, iou_thr: float) -> List[List[int]]:
    coords = arr[:, 3:7].astype(np.float32)
    iou_mat = _iou_matrix_xyxy(coords)
    adj = iou_mat > float(iou_thr)
    np.fill_diagonal(adj, True)
    return _connected_components(adj)


def _aggregate_confidence(conf_arr: np.ndarray, use_self_weighted: bool) -> float:
    conf_arr = conf_arr.astype(np.float32)
    if conf_arr.size == 0:
        return 0.0
    s1 = float(np.sum(conf_arr))
    if s1 <= 0.0:
        return 0.0
    if use_self_weighted:
        return float(np.sum(conf_arr * conf_arr) / max(s1, 1e-10))
    return float(np.mean(conf_arr))


def _rescale_confidence(conf: float, cluster_size: int, unique_models: int, total_models: int, use_unique: bool) -> float:
    if total_models <= 0:
        return float(conf)
    if use_unique:
        factor = min(float(unique_models), float(total_models)) / float(total_models)
    else:
        factor = min(float(cluster_size), float(total_models)) / float(total_models)
    return float(conf) * factor


def fuse_variant(
    preds_by_model: List[List[Dict]],
    variant: str,
    iou_thr: float = 0.55,
    weights: Optional[List[float]] = None,
    skip_box_thr: float = 0.0,
) -> List[Dict]:
    comps = variant_components(variant)
    use_transitive = comps["T_transitive_clustering"]
    use_self_weighted = comps["W_self_weighted_confidence"]
    use_unique_rescale = comps["U_unique_contributor_rescaling"]

    num_models = len(preds_by_model)
    if weights is None:
        weights = [1.0] * num_models
    if len(weights) != num_models:
        raise ValueError(f"weights length must equal number of models. Got {len(weights)} vs {num_models}")

    rows_by_cls: Dict[int, List[List[float]]] = {0: [], 1: []}

    # Row format: [label, weighted_conf, model_id, x1, y1, x2, y2]
    for model_idx, preds in enumerate(preds_by_model):
        model_weight = float(weights[model_idx])
        for p in preds:
            cls = int(p["cls"])
            conf = float(p["conf"])
            if cls not in CLASS_IDS:
                continue
            if conf < skip_box_thr:
                continue

            x1, y1, x2, y2 = p["box"]
            x1 = float(np.clip(x1, 0.0, 1.0))
            y1 = float(np.clip(y1, 0.0, 1.0))
            x2 = float(np.clip(x2, 0.0, 1.0))
            y2 = float(np.clip(y2, 0.0, 1.0))
            if x2 <= x1 or y2 <= y1:
                continue

            rows_by_cls[cls].append([
                float(cls),
                float(conf * model_weight),
                float(model_idx),
                x1, y1, x2, y2,
            ])

    fused_all = []

    for cls in CLASS_IDS:
        rows = rows_by_cls[cls]
        if not rows:
            continue

        arr = np.array(rows, dtype=np.float32)
        arr = arr[arr[:, 1].argsort()[::-1]]

        if use_transitive:
            clusters = _transitive_clusters(arr, iou_thr=iou_thr)
        else:
            clusters = _direct_clusters(arr, iou_thr=iou_thr)

        for inds in clusters:
            cluster = arr[np.array(inds, dtype=np.int32)]
            conf_arr = cluster[:, 1].astype(np.float32)
            model_ids = cluster[:, 2].astype(np.int32)

            fused_box = _fused_box_for_indices(arr, inds)
            raw_conf = _aggregate_confidence(conf_arr, use_self_weighted=use_self_weighted)
            unique_models = int(len(np.unique(model_ids)))
            final_conf = _rescale_confidence(
                raw_conf,
                cluster_size=len(inds),
                unique_models=unique_models,
                total_models=num_models,
                use_unique=use_unique_rescale,
            )

            fused_all.append({
                "box": fused_box,
                "conf": float(final_conf),
                "cls": int(cls),
                "contributors": int(unique_models),
                "cluster_size": int(len(inds)),
                "variant": variant,
                "variant_name": VARIANT_NAMES[variant],
            })

    return sorted(fused_all, key=lambda x: x["conf"], reverse=True)


# -------------------------------------------------------------------------
# CLIP loading and verification
# -------------------------------------------------------------------------
def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "model_state"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def clean_state_dict_keys(state):
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ["module.", "model.", "clip_model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


def load_weights_into_clip(model, weight_path: Path, device: str, label: str):
    weight_path = Path(weight_path)
    if not weight_path.is_file():
        raise FileNotFoundError(f"CLIP weight file not found: {weight_path}")

    if weight_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(weight_path), device=device)
    else:
        ckpt = torch.load(str(weight_path), map_location=device)
        state = extract_state_dict(ckpt)

    state = clean_state_dict_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CLIP] Loaded {label}: {weight_path}")
    print(f"[CLIP] {label} missing keys: {len(missing)}")
    print(f"[CLIP] {label} unexpected keys: {len(unexpected)}")
    if len(missing) > 20:
        print(f"[WARNING] Many missing keys after loading {label}. Check checkpoint compatibility.")
    return model


def load_clip_model(clip_ckpt: Path, clip_pretrained_path: Path, device: str, model_name: str = "ViT-B-32"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None)
    model = model.to(device)

    model = load_weights_into_clip(model, Path(clip_pretrained_path), device, label="local OpenCLIP base weights")
    model = load_weights_into_clip(model, Path(clip_ckpt), device, label="fine-tuned CLIP verifier checkpoint")

    tokenizer = open_clip.get_tokenizer(model_name)
    text_tokens = tokenizer(CLIP_PROMPTS).to(device)

    model.eval()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return model, preprocess, text_features


@torch.no_grad()
def clip_verify_crop(
    pil_img: Image.Image,
    norm_box: List[float],
    clip_model,
    clip_preprocess,
    text_features,
    device: str,
    pad_ratio: float = 0.05,
) -> Tuple[int, List[float], float]:
    img_w, img_h = pil_img.size
    pix_box = norm_to_pixel_box(norm_box, img_w, img_h)
    pix_box = pad_pixel_box(pix_box, img_w, img_h, pad_ratio=pad_ratio)

    crop = pil_img.crop(pix_box).convert("RGB")
    image_tensor = clip_preprocess(crop).unsqueeze(0).to(device)

    image_features = clip_model.encode_image(image_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    logit_scale = clip_model.logit_scale.exp()
    logits = logit_scale * image_features @ text_features.T
    probs = logits.softmax(dim=-1).squeeze(0).detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    entropy = -float(np.sum(probs * np.log(probs + 1e-12))) / math.log(len(probs))

    return pred_idx, probs.astype(float).tolist(), entropy


def attach_clip_scores_to_candidates(
    image_path: Path,
    candidates: List[Dict],
    min_conf: float,
    max_direct_conf: float,
    clip_model,
    clip_preprocess,
    text_features,
    device: str,
) -> List[Dict]:
    if len(candidates) == 0:
        return []

    pil_img = Image.open(image_path).convert("RGB")
    out = []

    for p in candidates:
        q = dict(p)
        # Cache CLIP only where it may be needed by filter-recovery threshold grids.
        if float(q["conf"]) >= float(min_conf) and float(q["conf"]) < float(max_direct_conf):
            pred_idx, probs, entropy = clip_verify_crop(
                pil_img,
                q["box"],
                clip_model,
                clip_preprocess,
                text_features,
                device,
                pad_ratio=0.05,
            )
            q["clip_pred"] = int(pred_idx)
            q["clip_probs"] = probs
            q["clip_entropy"] = float(entropy)
        out.append(q)

    return out


# -------------------------------------------------------------------------
# Filter-recovery and evaluation
# -------------------------------------------------------------------------
def apply_filter_recovery_from_cached(
    candidates: List[Dict],
    base_conf: float,
    direct_conf: float,
    recovery_min_conf: float,
    entropy_thr: float,
    duplicate_iou: float,
) -> List[Dict]:
    final_preds = []

    for p in candidates:
        conf = float(p["conf"])
        if conf < base_conf:
            continue

        if conf >= direct_conf:
            final_preds.append(dict(p))
            continue

        if "clip_pred" not in p or "clip_entropy" not in p:
            final_preds.append(dict(p))
            continue

        # Remove only confident non-weapon candidates.
        if int(p["clip_pred"]) == 2 and float(p["clip_entropy"]) <= entropy_thr:
            continue

        final_preds.append(dict(p))

    for p in candidates:
        conf = float(p["conf"])
        if not (recovery_min_conf <= conf < base_conf):
            continue
        if "clip_pred" not in p or "clip_entropy" not in p:
            continue
        if int(p["clip_pred"]) != int(p["cls"]):
            continue
        if float(p["clip_entropy"]) > entropy_thr:
            continue

        duplicate = False
        for kept in final_preds:
            if int(kept["cls"]) == int(p["cls"]) and box_iou(kept["box"], p["box"]) >= duplicate_iou:
                duplicate = True
                break

        if not duplicate:
            final_preds.append(dict(p))

    return sorted(final_preds, key=lambda x: x["conf"], reverse=True)


def match_counts(preds: List[Dict], gts: List[Dict], iou_thr: float = 0.50) -> Tuple[int, int, int]:
    matched = set()
    tp = 0
    fp = 0
    preds_sorted = sorted(preds, key=lambda x: x["conf"], reverse=True)

    for p in preds_sorted:
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gts):
            if j in matched:
                continue
            if int(p["cls"]) != int(g["cls"]):
                continue
            iou_val = box_iou(p["box"], g["box"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j

        if best_iou >= iou_thr and best_j >= 0:
            tp += 1
            matched.add(best_j)
        else:
            fp += 1

    fn = len(gts) - len(matched)
    return tp, fp, fn


def compute_basic_metrics(all_preds: Dict[str, List[Dict]], all_gts: Dict[str, List[Dict]], iou_thr: float = 0.50) -> Dict:
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_id, gts in all_gts.items():
        preds = all_preds.get(img_id, [])
        tp, fp, fn = match_counts(preds, gts, iou_thr=iou_thr)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    fdr = total_fp / (total_tp + total_fp + 1e-9)
    fnr = total_fn / (total_tp + total_fn + 1e-9)
    mde = 0.5 * (fdr + fnr)

    return {
        "TP": int(total_tp),
        "FP": int(total_fp),
        "FN": int(total_fn),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "FDR": float(fdr),
        "FNR": float(fnr),
        "MDE": float(mde),
    }


def ap_for_class(all_preds: Dict[str, List[Dict]], all_gts: Dict[str, List[Dict]], cls_id: int, iou_thr: float) -> Optional[float]:
    gt_by_img = {}
    n_gt = 0

    for img_id, gts in all_gts.items():
        cls_gts = [g for g in gts if int(g["cls"]) == cls_id]
        gt_by_img[img_id] = {"gts": cls_gts, "matched": np.zeros(len(cls_gts), dtype=bool)}
        n_gt += len(cls_gts)

    if n_gt == 0:
        return None

    pred_list = []
    for img_id, preds in all_preds.items():
        for p in preds:
            if int(p["cls"]) == cls_id:
                pred_list.append((img_id, float(p["conf"]), p["box"]))

    pred_list = sorted(pred_list, key=lambda x: x[1], reverse=True)
    if len(pred_list) == 0:
        return 0.0

    tp = np.zeros(len(pred_list), dtype=np.float32)
    fp = np.zeros(len(pred_list), dtype=np.float32)

    for i, (img_id, conf, box) in enumerate(pred_list):
        gt_info = gt_by_img.get(img_id, {"gts": [], "matched": np.zeros(0, dtype=bool)})
        gts = gt_info["gts"]
        best_iou = 0.0
        best_j = -1

        for j, g in enumerate(gts):
            if gt_info["matched"][j]:
                continue
            iou_val = box_iou(box, g["box"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j

        if best_iou >= iou_thr and best_j >= 0:
            tp[i] = 1.0
            gt_info["matched"][best_j] = True
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / (n_gt + 1e-9)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-9)

    ap = 0.0
    for r in np.linspace(0, 1, 101):
        prec_at_r = precisions[recalls >= r]
        ap += np.max(prec_at_r) if prec_at_r.size > 0 else 0.0
    ap /= 101.0

    return float(ap)


def compute_map(all_preds: Dict[str, List[Dict]], all_gts: Dict[str, List[Dict]]) -> Dict:
    ap50_values = []
    map5095_values = []
    iou_thresholds = np.arange(0.50, 0.96, 0.05)

    for cls_id in CLASS_IDS:
        ap50 = ap_for_class(all_preds, all_gts, cls_id, 0.50)
        if ap50 is not None:
            ap50_values.append(ap50)
        for t in iou_thresholds:
            ap = ap_for_class(all_preds, all_gts, cls_id, float(t))
            if ap is not None:
                map5095_values.append(ap)

    return {
        "mAP50": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "mAP50_95": float(np.mean(map5095_values)) if map5095_values else 0.0,
    }


def evaluate_predictions(all_preds: Dict[str, List[Dict]], all_gts: Dict[str, List[Dict]], fps: Optional[float] = None) -> Dict:
    out = {}
    out.update(compute_basic_metrics(all_preds, all_gts, iou_thr=0.50))
    out.update(compute_map(all_preds, all_gts))
    if fps is not None:
        out["FPS"] = float(fps)
    return round_dict(out)


def predictions_from_final_conf(candidates_by_img: Dict[str, List[Dict]], final_conf: float) -> Dict[str, List[Dict]]:
    return {
        img_id: [dict(p) for p in preds if float(p["conf"]) >= float(final_conf)]
        for img_id, preds in candidates_by_img.items()
    }


def predictions_from_filter_recovery(
    candidates_by_img: Dict[str, List[Dict]],
    base_conf: float,
    direct_conf: float,
    recovery_min_conf: float,
    entropy_thr: float,
    duplicate_iou: float,
) -> Dict[str, List[Dict]]:
    return {
        img_id: apply_filter_recovery_from_cached(
            preds,
            base_conf=base_conf,
            direct_conf=direct_conf,
            recovery_min_conf=recovery_min_conf,
            entropy_thr=entropy_thr,
            duplicate_iou=duplicate_iou,
        )
        for img_id, preds in candidates_by_img.items()
    }


def candidate_rank_tuple(metrics: Dict, metric_name: str) -> Tuple:
    if metric_name == "MDE":
        primary = -float(metrics["MDE"])
    else:
        primary = float(metrics[metric_name])
    return (
        primary,
        float(metrics.get("F1", 0.0)),
        float(metrics.get("Recall", 0.0)),
        -int(metrics.get("FP", 0)),
        -int(metrics.get("FN", 0)),
        float(metrics.get("Precision", 0.0)),
    )


def count_gts(all_gts: Dict[str, List[Dict]]) -> Dict:
    knife = 0
    pistol = 0
    for gts in all_gts.values():
        for g in gts:
            if int(g["cls"]) == 0:
                knife += 1
            elif int(g["cls"]) == 1:
                pistol += 1
    return {"gt_knife": knife, "gt_pistol_or_handgun": pistol, "gt_total": knife + pistol}


def write_rows_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(results: Dict, path: Path) -> None:
    rows = []
    for scope, scope_data in results["evaluation"].items():
        for method, metrics in scope_data["methods"].items():
            row = {"scope": scope, "method": method}
            row.update(metrics)
            rows.append(row)
    write_rows_csv(rows, path)


# -------------------------------------------------------------------------
# Dataset prediction generation for all variants or selected variants
# -------------------------------------------------------------------------
def generate_variant_candidates_for_dataset(
    dataset_name: str,
    image_paths: List[Path],
    image_root: Path,
    label_root: Path,
    y8x: YOLO,
    y8l: YOLO,
    args,
    variants_to_run: List[str],
    clip_bundle=None,
    clip_cache_min_conf: Optional[float] = None,
    clip_cache_max_direct_conf: Optional[float] = None,
) -> Tuple[Dict[str, Dict[str, List[Dict]]], Dict[str, List[Dict]], Dict]:
    all_gts: Dict[str, List[Dict]] = {}
    all_candidates_by_variant: Dict[str, Dict[str, List[Dict]]] = {v: {} for v in variants_to_run}

    device = args.device
    if clip_bundle is not None:
        clip_model, clip_preprocess, text_features = clip_bundle
    else:
        clip_model = clip_preprocess = text_features = None

    t_y8x = 0.0
    t_y8l = 0.0
    t_fusion_by_variant = {v: 0.0 for v in variants_to_run}
    t_clip_by_variant = {v: 0.0 for v in variants_to_run}

    start_all = time.perf_counter()

    for idx, img_path in enumerate(image_paths, 1):
        img_id = str(img_path.relative_to(image_root))

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        lbl_path = label_path_from_image(img_path, image_root, label_root)
        all_gts[img_id] = load_yolo_gt(lbl_path)

        t0 = time.perf_counter()
        p8x = yolo_predict_normxyxy(
            y8x, img_path, img_w, img_h,
            imgsz=args.imgsz,
            conf_th=args.det_conf,
            iou_th=args.det_iou,
            max_det=args.max_det,
            device=device,
        )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t_y8x += time.perf_counter() - t0

        t0 = time.perf_counter()
        p8l = yolo_predict_normxyxy(
            y8l, img_path, img_w, img_h,
            imgsz=args.imgsz,
            conf_th=args.det_conf,
            iou_th=args.det_iou,
            max_det=args.max_det,
            device=device,
        )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t_y8l += time.perf_counter() - t0

        for variant in variants_to_run:
            t0 = time.perf_counter()
            candidates = fuse_variant(
                [p8x, p8l],
                variant=variant,
                iou_thr=args.fusion_iou,
                weights=[1.0, 1.0],
                skip_box_thr=args.det_conf,
            )
            t_fusion_by_variant[variant] += time.perf_counter() - t0

            if clip_bundle is not None:
                min_conf = float(clip_cache_min_conf if clip_cache_min_conf is not None else args.min_recovery_conf_grid)
                max_direct = float(clip_cache_max_direct_conf if clip_cache_max_direct_conf is not None else args.max_direct_conf_grid)
                t0 = time.perf_counter()
                candidates = attach_clip_scores_to_candidates(
                    img_path,
                    candidates,
                    min_conf=min_conf,
                    max_direct_conf=max_direct,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                    text_features=text_features,
                    device=device,
                )
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                t_clip_by_variant[variant] += time.perf_counter() - t0

            all_candidates_by_variant[variant][img_id] = candidates

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"[INFO] {dataset_name}: processed {idx}/{len(image_paths)} images")

    total_time = time.perf_counter() - start_all
    n = len(image_paths)

    timing = {
        "images": n,
        "yolov8x_seconds": t_y8x,
        "yolov8l_seconds": t_y8l,
        "fusion_seconds_by_variant": t_fusion_by_variant,
        "clip_seconds_by_variant": t_clip_by_variant,
        "total_seconds": total_time,
        "fusion_only_fps_approx_by_variant": {
            v: n / max(t_y8x + t_y8l + t_fusion_by_variant[v], 1e-9) for v in variants_to_run
        },
        "fusion_clip_fps_approx_by_variant": {
            v: n / max(t_y8x + t_y8l + t_fusion_by_variant[v] + t_clip_by_variant[v], 1e-9) for v in variants_to_run
        },
    }

    return all_candidates_by_variant, all_gts, timing


# -------------------------------------------------------------------------
# Validation search
# -------------------------------------------------------------------------
def search_fusion_only_all_variants(
    candidates_by_variant: Dict[str, Dict[str, List[Dict]]],
    all_gts: Dict[str, List[Dict]],
    final_conf_grid: List[float],
    selection_metric: str,
) -> Tuple[Dict, Dict[str, Dict], List[Dict]]:
    rows = []
    best_global = None
    best_global_rank = None
    best_by_variant: Dict[str, Dict] = {}

    for variant in VARIANTS:
        best_variant = None
        best_variant_rank = None
        cands = candidates_by_variant[variant]
        comps = variant_components(variant)

        for final_conf in final_conf_grid:
            preds = predictions_from_final_conf(cands, final_conf)
            metrics = evaluate_predictions(preds, all_gts)
            row = {
                "setting_type": "fusion_only",
                "variant": variant,
                "variant_name": VARIANT_NAMES[variant],
                **comps,
                "final_conf": float(final_conf),
            }
            row.update(metrics)
            rows.append(row)

            rank = candidate_rank_tuple(metrics, selection_metric)
            if best_variant_rank is None or rank > best_variant_rank:
                best_variant_rank = rank
                best_variant = row
            if best_global_rank is None or rank > best_global_rank:
                best_global_rank = rank
                best_global = row

        best_by_variant[variant] = best_variant

    return best_global, best_by_variant, rows


def search_clip_fr_all_variants(
    candidates_by_variant: Dict[str, Dict[str, List[Dict]]],
    all_gts: Dict[str, List[Dict]],
    base_conf_grid: List[float],
    direct_conf_grid: List[float],
    recovery_min_conf_grid: List[float],
    entropy_thr_grid: List[float],
    duplicate_iou: float,
    selection_metric: str,
) -> Tuple[Dict, Dict[str, Dict], List[Dict]]:
    rows = []
    best_global = None
    best_global_rank = None
    best_by_variant: Dict[str, Dict] = {}

    for variant in VARIANTS:
        best_variant = None
        best_variant_rank = None
        cands = candidates_by_variant[variant]
        comps = variant_components(variant)

        for base_conf in base_conf_grid:
            for direct_conf in direct_conf_grid:
                if direct_conf < base_conf:
                    continue
                for recovery_min_conf in recovery_min_conf_grid:
                    if recovery_min_conf >= base_conf:
                        continue
                    for entropy_thr in entropy_thr_grid:
                        preds = predictions_from_filter_recovery(
                            cands,
                            base_conf=base_conf,
                            direct_conf=direct_conf,
                            recovery_min_conf=recovery_min_conf,
                            entropy_thr=entropy_thr,
                            duplicate_iou=duplicate_iou,
                        )
                        metrics = evaluate_predictions(preds, all_gts)
                        row = {
                            "setting_type": "fusion_plus_clip_fr",
                            "variant": variant,
                            "variant_name": VARIANT_NAMES[variant],
                            **comps,
                            "base_conf": float(base_conf),
                            "direct_conf": float(direct_conf),
                            "recovery_min_conf": float(recovery_min_conf),
                            "entropy_thr": float(entropy_thr),
                            "duplicate_iou": float(duplicate_iou),
                        }
                        row.update(metrics)
                        rows.append(row)

                        rank = candidate_rank_tuple(metrics, selection_metric)
                        if best_variant_rank is None or rank > best_variant_rank:
                            best_variant_rank = rank
                            best_variant = row
                        if best_global_rank is None or rank > best_global_rank:
                            best_global_rank = rank
                            best_global = row

        best_by_variant[variant] = best_variant

    return best_global, best_by_variant, rows


def evaluate_selected_for_dataset(
    candidates_by_variant: Dict[str, Dict[str, List[Dict]]],
    all_gts: Dict[str, List[Dict]],
    selected_fusion_global: Dict,
    selected_clip_global: Dict,
    best_fusion_by_variant: Dict[str, Dict],
    timing: Dict,
) -> Dict[str, Dict]:
    methods = {}

    # Global best fusion-only selected on validation.
    v_fg = selected_fusion_global["variant"]
    preds_fg = predictions_from_final_conf(
        candidates_by_variant[v_fg],
        final_conf=selected_fusion_global["final_conf"],
    )
    fps_fg = timing["fusion_only_fps_approx_by_variant"].get(v_fg)
    methods[f"GlobalBestFusionOnly_{v_fg}_{VARIANT_NAMES[v_fg]}"] = evaluate_predictions(preds_fg, all_gts, fps=fps_fg)

    # Fusion-only counterpart for the globally selected CLIP-FR variant.
    v_clip = selected_clip_global["variant"]
    fusion_counterpart = best_fusion_by_variant[v_clip]
    preds_counter = predictions_from_final_conf(
        candidates_by_variant[v_clip],
        final_conf=fusion_counterpart["final_conf"],
    )
    fps_counter = timing["fusion_only_fps_approx_by_variant"].get(v_clip)
    methods[f"FusionOnly_for_SelectedCLIPVariant_{v_clip}_{VARIANT_NAMES[v_clip]}"] = evaluate_predictions(
        preds_counter,
        all_gts,
        fps=fps_counter,
    )

    # Global best fusion + CLIP-FR selected on validation.
    preds_clip = predictions_from_filter_recovery(
        candidates_by_variant[v_clip],
        base_conf=selected_clip_global["base_conf"],
        direct_conf=selected_clip_global["direct_conf"],
        recovery_min_conf=selected_clip_global["recovery_min_conf"],
        entropy_thr=selected_clip_global["entropy_thr"],
        duplicate_iou=selected_clip_global["duplicate_iou"],
    )
    fps_clip = timing["fusion_clip_fps_approx_by_variant"].get(v_clip)
    methods[f"GlobalBestFusionPlusCLIP_FR_{v_clip}_{VARIANT_NAMES[v_clip]}"] = evaluate_predictions(
        preds_clip,
        all_gts,
        fps=fps_clip,
    )

    return methods


# -------------------------------------------------------------------------
# CLI and main
# -------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser("YOLO-only CC-WBF component ablation + CLIP-FR evaluation")

    ap.add_argument("--dataset_root", type=str, required=True, help="Main dataset root with images/<split> and labels/<split>")
    ap.add_argument("--valid_split", type=str, default="valid")
    ap.add_argument("--test_split", type=str, default="test")

    ap.add_argument("--external_images", type=str, default="", help="Optional external dataset image folder")
    ap.add_argument("--external_labels", type=str, default="", help="Optional external dataset label folder")
    ap.add_argument("--external_name", type=str, default="external_zero_shot")

    ap.add_argument("--y8x", type=str, required=True, help="Frozen YOLOv8x detector checkpoint")
    ap.add_argument("--y8l", type=str, required=True, help="Frozen YOLOv8l detector checkpoint")
    ap.add_argument("--clip_ckpt", type=str, required=True, help="Fine-tuned CLIP verifier checkpoint")
    ap.add_argument("--clip_pretrained_path", type=str, required=True, help="Local OpenCLIP safetensors/pt/pth/bin file")

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=640)

    # Frozen detector settings from the current paper.
    ap.add_argument("--det_conf", type=float, default=0.001)
    ap.add_argument("--det_iou", type=float, default=0.70)
    ap.add_argument("--max_det", type=int, default=300)

    # Shared fusion IoU for all component variants.
    ap.add_argument("--fusion_iou", type=float, default=0.55)

    # Validation threshold search grids.
    ap.add_argument("--final_conf_grid", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    ap.add_argument("--base_conf_grid", type=str, default="0.20,0.25,0.30,0.35")
    ap.add_argument("--direct_conf_grid", type=str, default="0.45,0.50,0.55,0.60")
    ap.add_argument("--recovery_min_conf_grid", type=str, default="0.10,0.15,0.20,0.25")
    ap.add_argument("--entropy_thr_grid", type=str, default="0.05,0.10,0.20,0.30")
    ap.add_argument("--duplicate_iou", type=float, default=0.50)
    ap.add_argument("--selection_metric", choices=["F1", "Recall", "Precision", "MDE"], default="F1")

    ap.add_argument("--clip_model", type=str, default="ViT-B-32")

    return ap.parse_args()


def main():
    args = parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[ERROR] CUDA requested but not available.")

    torch.set_grad_enabled(False)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    for p in [args.dataset_root, args.y8x, args.y8l, args.clip_ckpt, args.clip_pretrained_path]:
        pp = Path(p)
        if not pp.exists():
            raise FileNotFoundError(f"Path not found: {pp}")

    print(f"[INFO] Device: {args.device}")
    print("[INFO] Loading frozen YOLO detectors...")
    y8x = YOLO(args.y8x)
    y8l = YOLO(args.y8l)

    print("[INFO] Loading frozen CLIP verifier...")
    clip_bundle = load_clip_model(
        clip_ckpt=Path(args.clip_ckpt),
        clip_pretrained_path=Path(args.clip_pretrained_path),
        device=args.device,
        model_name=args.clip_model,
    )

    final_conf_grid = parse_float_list(args.final_conf_grid)
    base_conf_grid = parse_float_list(args.base_conf_grid)
    direct_conf_grid = parse_float_list(args.direct_conf_grid)
    recovery_min_conf_grid = parse_float_list(args.recovery_min_conf_grid)
    entropy_thr_grid = parse_float_list(args.entropy_thr_grid)

    args.min_recovery_conf_grid = min(recovery_min_conf_grid)
    args.max_direct_conf_grid = max(direct_conf_grid)

    dataset_root = Path(args.dataset_root)

    print("[INFO] Loading main validation split...")
    valid_imgs, valid_img_root, valid_lbl_root = load_split_from_root(dataset_root, args.valid_split)
    print(f"[INFO] Main validation images: {len(valid_imgs)}")

    print("[INFO] Generating validation candidates for all 8 variants and caching CLIP scores...")
    valid_candidates_by_variant, valid_gts, valid_timing = generate_variant_candidates_for_dataset(
        dataset_name="main_validation",
        image_paths=valid_imgs,
        image_root=valid_img_root,
        label_root=valid_lbl_root,
        y8x=y8x,
        y8l=y8l,
        args=args,
        variants_to_run=VARIANTS,
        clip_bundle=clip_bundle,
        clip_cache_min_conf=args.min_recovery_conf_grid,
        clip_cache_max_direct_conf=args.max_direct_conf_grid,
    )

    print("[INFO] Searching all fusion-only component variants on validation split...")
    best_fusion_global, best_fusion_by_variant, fusion_rows = search_fusion_only_all_variants(
        valid_candidates_by_variant,
        valid_gts,
        final_conf_grid=final_conf_grid,
        selection_metric=args.selection_metric,
    )

    print("[INFO] Searching all fusion + CLIP-FR component variants on validation split...")
    best_clip_global, best_clip_by_variant, clip_rows = search_clip_fr_all_variants(
        valid_candidates_by_variant,
        valid_gts,
        base_conf_grid=base_conf_grid,
        direct_conf_grid=direct_conf_grid,
        recovery_min_conf_grid=recovery_min_conf_grid,
        entropy_thr_grid=entropy_thr_grid,
        duplicate_iou=args.duplicate_iou,
        selection_metric=args.selection_metric,
    )

    best_per_variant_rows = []
    for v in VARIANTS:
        best_per_variant_rows.append({"selected_for": "fusion_only", **best_fusion_by_variant[v]})
        best_per_variant_rows.append({"selected_for": "fusion_plus_clip_fr", **best_clip_by_variant[v]})

    selected_clip_variant = best_clip_global["variant"]
    selected_thresholds = {
        "selection_metric": args.selection_metric,
        "selected_on": "main_validation_split_only",
        "variant_definitions": {
            "T": "transitive clustering",
            "W": "self-weighted confidence aggregation",
            "U": "unique-contributor confidence rescaling",
            "bit_order": "T W U",
            "variants": VARIANT_NAMES,
        },
        "global_best_fusion_only": best_fusion_global,
        "global_best_fusion_plus_clip_fr": best_clip_global,
        "fusion_only_counterpart_for_selected_clip_variant": best_fusion_by_variant[selected_clip_variant],
        "best_fusion_only_by_variant": best_fusion_by_variant,
        "best_fusion_plus_clip_fr_by_variant": best_clip_by_variant,
    }

    print("\n========== Selected validation settings ==========")
    print(json.dumps({
        "global_best_fusion_only": best_fusion_global,
        "fusion_only_counterpart_for_selected_clip_variant": best_fusion_by_variant[selected_clip_variant],
        "global_best_fusion_plus_clip_fr": best_clip_global,
    }, indent=2))

    (out_dir / "selected_component_variant_thresholds.json").write_text(json.dumps(selected_thresholds, indent=2), encoding="utf-8")
    write_rows_csv(fusion_rows + clip_rows, out_dir / "validation_all_component_threshold_search.csv")
    write_rows_csv(best_per_variant_rows, out_dir / "validation_best_per_variant.csv")

    # Validation summary using selected settings.
    valid_eval_methods = evaluate_selected_for_dataset(
        valid_candidates_by_variant,
        valid_gts,
        selected_fusion_global=best_fusion_global,
        selected_clip_global=best_clip_global,
        best_fusion_by_variant=best_fusion_by_variant,
        timing=valid_timing,
    )

    # Test split: only run the globally selected fusion-only variant and the selected CLIP variant.
    selected_variants_for_final = sorted(set([best_fusion_global["variant"], selected_clip_variant]))

    print("[INFO] Loading main test split...")
    test_imgs, test_img_root, test_lbl_root = load_split_from_root(dataset_root, args.test_split)
    print(f"[INFO] Main test images: {len(test_imgs)}")

    print(f"[INFO] Generating test candidates for selected variants only: {selected_variants_for_final}")
    test_candidates_by_variant, test_gts, test_timing = generate_variant_candidates_for_dataset(
        dataset_name="main_test",
        image_paths=test_imgs,
        image_root=test_img_root,
        label_root=test_lbl_root,
        y8x=y8x,
        y8l=y8l,
        args=args,
        variants_to_run=selected_variants_for_final,
        clip_bundle=clip_bundle,
        clip_cache_min_conf=args.min_recovery_conf_grid,
        clip_cache_max_direct_conf=args.max_direct_conf_grid,
    )

    test_eval_methods = evaluate_selected_for_dataset(
        test_candidates_by_variant,
        test_gts,
        selected_fusion_global=best_fusion_global,
        selected_clip_global=best_clip_global,
        best_fusion_by_variant=best_fusion_by_variant,
        timing=test_timing,
    )

    results = {
        "pipeline": "YOLOv8x + YOLOv8l + 8 CC-WBF component variants + optional CLIP filter-recovery",
        "note": "Fusion component variant and thresholds are selected only on the main validation split, then frozen for test and external evaluation.",
        "class_mapping": {"0": "knife", "1": "pistol or handgun treated as pistol"},
        "model_paths": {
            "yolov8x": args.y8x,
            "yolov8l": args.y8l,
            "clip_ckpt": args.clip_ckpt,
            "clip_pretrained_path": args.clip_pretrained_path,
        },
        "fixed_detector_and_fusion_settings": {
            "det_conf": args.det_conf,
            "det_iou": args.det_iou,
            "max_det": args.max_det,
            "fusion_iou": args.fusion_iou,
            "duplicate_iou": args.duplicate_iou,
            "clip_model": args.clip_model,
            "clip_prompts": CLIP_PROMPTS,
        },
        "threshold_search_grids": {
            "final_conf_grid": final_conf_grid,
            "base_conf_grid": base_conf_grid,
            "direct_conf_grid": direct_conf_grid,
            "recovery_min_conf_grid": recovery_min_conf_grid,
            "entropy_thr_grid": entropy_thr_grid,
        },
        "selected_settings": selected_thresholds,
        "timing_seconds": {
            "main_validation": round_dict(valid_timing),
            "main_test": round_dict(test_timing),
        },
        "evaluation": {
            "main_validation_selected_from_validation": {
                "images": len(valid_imgs),
                **count_gts(valid_gts),
                "methods": valid_eval_methods,
            },
            "main_test_frozen_selected_settings": {
                "images": len(test_imgs),
                **count_gts(test_gts),
                "methods": test_eval_methods,
            },
        },
    }

    # Optional external zero-shot evaluation with the same frozen selected settings.
    if args.external_images and args.external_labels:
        ext_imgs, ext_img_root, ext_lbl_root = load_dataset_from_images_labels(Path(args.external_images), Path(args.external_labels))
        print(f"[INFO] Generating external zero-shot candidates for selected variants only: {selected_variants_for_final}")
        ext_candidates_by_variant, ext_gts, ext_timing = generate_variant_candidates_for_dataset(
            dataset_name=args.external_name,
            image_paths=ext_imgs,
            image_root=ext_img_root,
            label_root=ext_lbl_root,
            y8x=y8x,
            y8l=y8l,
            args=args,
            variants_to_run=selected_variants_for_final,
            clip_bundle=clip_bundle,
            clip_cache_min_conf=args.min_recovery_conf_grid,
            clip_cache_max_direct_conf=args.max_direct_conf_grid,
        )

        ext_eval_methods = evaluate_selected_for_dataset(
            ext_candidates_by_variant,
            ext_gts,
            selected_fusion_global=best_fusion_global,
            selected_clip_global=best_clip_global,
            best_fusion_by_variant=best_fusion_by_variant,
            timing=ext_timing,
        )

        results["timing_seconds"][args.external_name] = round_dict(ext_timing)
        results["evaluation"][f"{args.external_name}_frozen_selected_settings"] = {
            "images": len(ext_imgs),
            **count_gts(ext_gts),
            "methods": ext_eval_methods,
        }

    print("\n========== Final Component-Ablation Results ==========")
    print(json.dumps(results, indent=2))

    out_json = out_dir / "component_ablation_ccwbf_clip_results.json"
    out_csv = out_dir / "component_ablation_ccwbf_clip_summary.csv"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary_csv(results, out_csv)

    print(f"\n[INFO] Saved selected settings to: {out_dir / 'selected_component_variant_thresholds.json'}")
    print(f"[INFO] Saved validation search to: {out_dir / 'validation_all_component_threshold_search.csv'}")
    print(f"[INFO] Saved best per variant to: {out_dir / 'validation_best_per_variant.csv'}")
    print(f"[INFO] Saved JSON results to: {out_json}")
    print(f"[INFO] Saved CSV summary to: {out_csv}")


if __name__ == "__main__":
    main()
