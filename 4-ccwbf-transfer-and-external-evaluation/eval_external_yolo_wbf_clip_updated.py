import os
import csv
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import open_clip


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# External dataset:
# 0 = knife
# 1 = handgun
# In this evaluation, handgun is treated as pistol to match the main paper labels.
CLASS_NAMES = {
    0: "knife",
    1: "pistol"
}

CLIP_PROMPTS = [
    "a cropped image of a knife",
    "a cropped image of a pistol",
    "a cropped image of a non-weapon object"
]


def xywhn_to_xyxy(box, img_w, img_h):
    x, y, w, h = box
    x1 = (x - w / 2.0) * img_w
    y1 = (y - h / 2.0) * img_h
    x2 = (x + w / 2.0) * img_w
    y2 = (y + h / 2.0) * img_h
    return [x1, y1, x2, y2]


def box_iou(a, b):
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
    return inter / union


def clip_box_xyxy(box, img_w, img_h):
    x1, y1, x2, y2 = box

    x1 = max(0, min(img_w - 1, int(round(x1))))
    y1 = max(0, min(img_h - 1, int(round(y1))))
    x2 = max(0, min(img_w, int(round(x2))))
    y2 = max(0, min(img_h, int(round(y2))))

    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    return [x1, y1, x2, y2]


def pad_box(box, img_w, img_h, pad_ratio=0.05):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad_ratio
    py = bh * pad_ratio
    return clip_box_xyxy([x1 - px, y1 - py, x2 + px, y2 + py], img_w, img_h)


def load_gt(label_path: Path, img_w: int, img_h: int):
    gt = []

    if not label_path.exists():
        return gt

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))

            if cls not in [0, 1]:
                continue

            vals = list(map(float, parts[1:5]))
            xyxy = xywhn_to_xyxy(vals, img_w, img_h)

            gt.append({
                "cls": cls,
                "box": xyxy
            })

    return gt


def find_images(image_root: Path):
    image_paths = []

    for p in image_root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            image_paths.append(p)

    return sorted(image_paths)


def label_path_from_image(image_path: Path, image_root: Path, label_root: Path):
    rel = image_path.relative_to(image_root)
    return label_root / rel.with_suffix(".txt")


def get_split_name(image_path: Path, image_root: Path):
    rel = image_path.relative_to(image_root)
    parts = rel.parts

    if len(parts) == 0:
        return "full"

    first = parts[0].lower()

    if first in ["train", "training"]:
        return "train"
    if first in ["valid", "val", "validation"]:
        return "validation"
    if first in ["test", "testing"]:
        return "test"

    return "full"


def yolo_predict(model, image_path, imgsz=640, conf=0.001, iou=0.70, max_det=300):
    result = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False
    )[0]

    preds = []

    if result.boxes is None or len(result.boxes) == 0:
        return preds

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy().astype(int)

    for b, c, k in zip(boxes, confs, clss):
        if k not in [0, 1]:
            continue

        preds.append({
            "box": b.astype(float).tolist(),
            "conf": float(c),
            "cls": int(k)
        })

    return preds


def wbf_fuse(preds_by_model, iou_thr=0.55, num_models=2):
    all_preds = []

    for model_idx, preds in enumerate(preds_by_model):
        for p in preds:
            q = dict(p)
            q["model_idx"] = model_idx
            all_preds.append(q)

    fused_all = []

    for cls in [0, 1]:
        cls_preds = [p for p in all_preds if p["cls"] == cls]
        cls_preds = sorted(cls_preds, key=lambda x: x["conf"], reverse=True)

        clusters = []

        for p in cls_preds:
            assigned = False
            best_iou = 0.0
            best_idx = -1

            for idx, cluster in enumerate(clusters):
                iou_val = box_iou(p["box"], cluster["fused_box"])

                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = idx

            if best_iou >= iou_thr and best_idx >= 0:
                cluster = clusters[best_idx]

                cluster["boxes"].append(p["box"])
                cluster["confs"].append(p["conf"])
                cluster["model_idxs"].append(p["model_idx"])

                conf_arr = np.array(cluster["confs"], dtype=np.float64)
                box_arr = np.array(cluster["boxes"], dtype=np.float64)

                weighted_box = (box_arr * conf_arr[:, None]).sum(axis=0) / (conf_arr.sum() + 1e-9)
                cluster["fused_box"] = weighted_box.tolist()

                assigned = True

            if not assigned:
                clusters.append({
                    "boxes": [p["box"]],
                    "confs": [p["conf"]],
                    "model_idxs": [p["model_idx"]],
                    "fused_box": p["box"],
                    "cls": cls
                })

        for cluster in clusters:
            unique_models = len(set(cluster["model_idxs"]))
            avg_conf = float(np.mean(cluster["confs"]))

            # Same frozen setting used in the paper:
            # confidence averaging with min-style contributor rescaling.
            rescale = min(unique_models, num_models) / float(num_models)
            fused_conf = avg_conf * rescale

            fused_all.append({
                "box": cluster["fused_box"],
                "conf": fused_conf,
                "cls": cls
            })

    fused_all = sorted(fused_all, key=lambda x: x["conf"], reverse=True)
    return fused_all


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


def resolve_weight_file(path_str):
    """Return a local OpenCLIP weight file.

    The path may be either the exact weight file or a folder containing the file.
    Supported extensions include .safetensors because your ViT-B-32_laion2b_e16
    weights are stored as open_clip_model.safetensors.
    """
    path = Path(path_str)

    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"Pretrained OpenCLIP path does not exist: {path}")

    candidates = []
    for ext in ["*.safetensors", "*.pt", "*.pth", "*.bin"]:
        candidates.extend(list(path.rglob(ext)))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No .safetensors, .pt, .pth, or .bin weight file found inside: {path}"
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)

    print("[CLIP] Candidate local pretrained files:")
    for p in candidates[:5]:
        print(f"       {p}  size={p.stat().st_size}")

    return candidates[0]


def load_weights_into_model(model, weight_path, device, label):
    """Load PyTorch or safetensors weights into an OpenCLIP model."""
    weight_path = Path(weight_path)

    if not weight_path.exists():
        raise FileNotFoundError(f"{label} path not found: {weight_path}")

    if weight_path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "The file is a .safetensors checkpoint, but safetensors is not installed. "
                "Install it with: pip install safetensors"
            ) from exc

        # safetensors accepts a device string such as 'cuda:0' or 'cpu'.
        state = load_file(str(weight_path), device=str(device))
    else:
        ckpt = torch.load(str(weight_path), map_location=device)
        state = extract_state_dict(ckpt)

    state = clean_state_dict_keys(state)

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[CLIP] Loaded {label}: {weight_path}")
    print(f"[CLIP] {label} missing keys: {len(missing)}")
    print(f"[CLIP] {label} unexpected keys: {len(unexpected)}")

    if unexpected:
        print(f"[CLIP] First unexpected keys for {label}: {unexpected[:5]}")
    if missing:
        print(f"[CLIP] First missing keys for {label}: {missing[:5]}")

    return model, missing, unexpected


def load_clip_model(
    ckpt_path,
    device,
    model_name="ViT-B-32",
    local_pretrained_path="/home/inspur/ishrai/pretrained_openclip/ViT-B-32_laion2b_e16/open_clip_model.safetensors"
):
    # Create architecture only. No internet download is used.
    # The expected warning "No pretrained weights loaded" is normal here.
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=None
    )

    model = model.to(device)

    # Load local LAION/OpenCLIP base weights first.
    local_weight_file = resolve_weight_file(local_pretrained_path)
    model, base_missing, base_unexpected = load_weights_into_model(
        model,
        local_weight_file,
        device,
        label="local OpenCLIP base weights"
    )

    # Load the fine-tuned verifier checkpoint on top of the base model.
    model, ft_missing, ft_unexpected = load_weights_into_model(
        model,
        ckpt_path,
        device,
        label="fine-tuned CLIP verifier checkpoint"
    )

    if len(ft_unexpected) > 50:
        print("[WARNING] Many unexpected keys were found in the fine-tuned checkpoint.")
        print("[WARNING] Check whether the checkpoint was saved from a wrapper model.")

    tokenizer = open_clip.get_tokenizer(model_name)
    text_tokens = tokenizer(CLIP_PROMPTS).to(device)

    model.eval()

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return model, preprocess, text_features


@torch.no_grad()
def clip_verify_crop(pil_img, box, clip_model, clip_preprocess, text_features, device):
    img_w, img_h = pil_img.size

    crop_box = pad_box(box, img_w, img_h, pad_ratio=0.05)
    crop = pil_img.crop(crop_box).convert("RGB")

    image_tensor = clip_preprocess(crop).unsqueeze(0).to(device)

    image_features = clip_model.encode_image(image_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    logit_scale = clip_model.logit_scale.exp()
    logits = logit_scale * image_features @ text_features.T

    probs = logits.softmax(dim=-1).squeeze(0).detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    entropy = -float(np.sum(probs * np.log(probs + 1e-12))) / math.log(len(probs))

    return pred_idx, probs, entropy


def apply_filter_recovery(
    image_path,
    candidates,
    clip_model,
    clip_preprocess,
    text_features,
    device,
    base_conf=0.25,
    direct_conf=0.50,
    recovery_min_conf=0.15,
    entropy_thr=0.20,
    duplicate_iou=0.50
):
    pil_img = Image.open(image_path).convert("RGB")

    final_preds = []

    # Filtering stage for WBF candidates above 0.25.
    for p in candidates:
        if p["conf"] < base_conf:
            continue

        # Directly accept high-confidence WBF detections.
        if p["conf"] >= direct_conf:
            final_preds.append(dict(p))
            continue

        pred_idx, probs, entropy = clip_verify_crop(
            pil_img,
            p["box"],
            clip_model,
            clip_preprocess,
            text_features,
            device
        )

        # CLIP class indices:
        # 0 = knife
        # 1 = pistol
        # 2 = non_weapon
        if pred_idx == 2 and entropy <= entropy_thr:
            continue

        q = dict(p)
        q["clip_pred"] = int(pred_idx)
        q["clip_entropy"] = float(entropy)
        q["clip_probs"] = probs.tolist()
        final_preds.append(q)

    # Recovery stage for lower-confidence WBF candidates.
    recovery_candidates = [
        p for p in candidates
        if recovery_min_conf <= p["conf"] < base_conf
    ]

    for p in recovery_candidates:
        pred_idx, probs, entropy = clip_verify_crop(
            pil_img,
            p["box"],
            clip_model,
            clip_preprocess,
            text_features,
            device
        )

        # Recover only when CLIP confidently agrees with the detector class.
        if pred_idx != p["cls"]:
            continue

        if entropy > entropy_thr:
            continue

        duplicate = False

        for kept in final_preds:
            if kept["cls"] == p["cls"] and box_iou(kept["box"], p["box"]) >= duplicate_iou:
                duplicate = True
                break

        if duplicate:
            continue

        q = dict(p)
        q["clip_pred"] = int(pred_idx)
        q["clip_entropy"] = float(entropy)
        q["clip_probs"] = probs.tolist()
        final_preds.append(q)

    final_preds = sorted(final_preds, key=lambda x: x["conf"], reverse=True)
    return final_preds


def match_counts(preds, gts, iou_thr=0.50):
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

            if p["cls"] != g["cls"]:
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


def compute_basic_metrics(all_preds, all_gts, iou_thr=0.50):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_id in all_gts.keys():
        preds = all_preds.get(img_id, [])
        gts = all_gts.get(img_id, [])

        tp, fp, fn = match_counts(preds, gts, iou_thr=iou_thr)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    fdr = total_fp / (total_tp + total_fp + 1e-9)

    return {
        "TP": int(total_tp),
        "FP": int(total_fp),
        "FN": int(total_fn),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "FDR": float(fdr)
    }


def ap_for_class(all_preds, all_gts, cls_id, iou_thr):
    gt_by_img = {}
    n_gt = 0

    for img_id, gts in all_gts.items():
        cls_gts = [g for g in gts if g["cls"] == cls_id]

        gt_by_img[img_id] = {
            "gts": cls_gts,
            "matched": np.zeros(len(cls_gts), dtype=bool)
        }

        n_gt += len(cls_gts)

    if n_gt == 0:
        return None

    pred_list = []

    for img_id, preds in all_preds.items():
        for p in preds:
            if p["cls"] == cls_id:
                pred_list.append((img_id, p["conf"], p["box"]))

    pred_list = sorted(pred_list, key=lambda x: x[1], reverse=True)

    if len(pred_list) == 0:
        return 0.0

    tp = np.zeros(len(pred_list))
    fp = np.zeros(len(pred_list))

    for i, (img_id, conf, box) in enumerate(pred_list):
        gt_info = gt_by_img.get(img_id, {"gts": [], "matched": []})
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
            tp[i] = 1
            gt_info["matched"][best_j] = True
        else:
            fp[i] = 1

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


def compute_map(all_preds, all_gts):
    ap50_values = []
    map5095_values = []

    iou_thresholds = np.arange(0.50, 0.96, 0.05)

    for cls_id in [0, 1]:
        ap50 = ap_for_class(all_preds, all_gts, cls_id, 0.50)

        if ap50 is not None:
            ap50_values.append(ap50)

        for t in iou_thresholds:
            ap = ap_for_class(all_preds, all_gts, cls_id, float(t))

            if ap is not None:
                map5095_values.append(ap)

    return {
        "mAP50": float(np.mean(ap50_values)) if ap50_values else 0.0,
        "mAP50_95": float(np.mean(map5095_values)) if map5095_values else 0.0
    }


def round_metrics(d):
    out = {}

    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, 4)
        else:
            out[k] = v

    return out


def evaluate_method(all_preds, all_gts, fps=None):
    basic = compute_basic_metrics(all_preds, all_gts, iou_thr=0.50)
    maps = compute_map(all_preds, all_gts)

    out = {}
    out.update(basic)
    out.update(maps)

    if fps is not None:
        out["FPS"] = float(fps)

    return round_metrics(out)


def filter_dict_by_split(data_dict, split_by_img, split_name):
    return {
        img_id: value
        for img_id, value in data_dict.items()
        if split_by_img.get(img_id, "full") == split_name
    }


def count_gts(all_gts):
    c0 = 0
    c1 = 0

    for gts in all_gts.values():
        for g in gts:
            if g["cls"] == 0:
                c0 += 1
            elif g["cls"] == 1:
                c1 += 1

    return c0, c1


def write_csv_summary(results, csv_path):
    rows = []

    for scope_name, scope_data in results["evaluation"].items():
        for method_name, metrics in scope_data["methods"].items():
            row = {
                "scope": scope_name,
                "method": method_name
            }

            row.update(metrics)
            rows.append(row)

    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images",
        type=str,
        default="/home/inspur/ishrai/yolo_mmd_from_rq1/images"
    )

    parser.add_argument(
        "--labels",
        type=str,
        default="/home/inspur/ishrai/yolo_mmd_from_rq1/labels"
    )

    parser.add_argument(
        "--y8x",
        type=str,
        required=True,
        help="Path to frozen YOLOv8x detector best.pt"
    )

    parser.add_argument(
        "--y8l",
        type=str,
        required=True,
        help="Path to frozen YOLOv8l detector best.pt"
    )

    parser.add_argument(
        "--clip_ckpt",
        type=str,
        required=True,
        help="Path to fine-tuned CLIP verifier checkpoint"
    )

    parser.add_argument(
        "--clip_pretrained_path",
        type=str,
        default="/home/inspur/ishrai/pretrained_openclip/ViT-B-32_laion2b_e16/open_clip_model.safetensors",
        help="Local OpenCLIP ViT-B-32 LAION pretrained weight folder or file"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/inspur/ishrai/external_yolo_mmd_eval"
    )

    parser.add_argument("--imgsz", type=int, default=640)

    # Frozen detector settings.
    parser.add_argument("--det_conf", type=float, default=0.001)
    parser.add_argument("--det_iou", type=float, default=0.70)
    parser.add_argument("--max_det", type=int, default=300)

    # Frozen WBF and final decision settings.
    parser.add_argument("--wbf_iou", type=float, default=0.55)
    parser.add_argument("--wbf_base_conf", type=float, default=0.25)
    parser.add_argument("--direct_conf", type=float, default=0.50)
    parser.add_argument("--recovery_min_conf", type=float, default=0.15)
    parser.add_argument("--entropy_thr", type=float, default=0.20)
    parser.add_argument("--duplicate_iou", type=float, default=0.50)

    parser.add_argument("--clip_model", type=str, default="ViT-B-32")

    args = parser.parse_args()

    image_root = Path(args.images)
    label_root = Path(args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for required_path in [image_root, label_root, Path(args.y8x), Path(args.y8l), Path(args.clip_ckpt)]:
        if not required_path.exists():
            raise FileNotFoundError(f"Path not found: {required_path}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    image_paths = find_images(image_root)
    print(f"[INFO] Found images: {len(image_paths)}")

    print("[INFO] Loading frozen YOLO detectors...")
    y8x = YOLO(args.y8x)
    y8l = YOLO(args.y8l)

    print("[INFO] Loading frozen CLIP verifier...")
    clip_model, clip_preprocess, text_features = load_clip_model(
        ckpt_path=args.clip_ckpt,
        device=device,
        model_name=args.clip_model,
        local_pretrained_path=args.clip_pretrained_path
    )

    all_gts = {}
    split_by_img = {}

    preds_y8x = {}
    preds_y8l = {}
    preds_wbf = {}
    preds_wbf_clip = {}

    gt_class_counts = {0: 0, 1: 0}

    t_y8x = 0.0
    t_y8l = 0.0
    t_wbf = 0.0
    t_clip = 0.0

    start_all = time.perf_counter()

    for idx, img_path in enumerate(image_paths, 1):
        img_id = str(img_path.relative_to(image_root))
        split_name = get_split_name(img_path, image_root)
        split_by_img[img_id] = split_name

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        lbl_path = label_path_from_image(img_path, image_root, label_root)
        gts = load_gt(lbl_path, img_w, img_h)

        for g in gts:
            gt_class_counts[g["cls"]] += 1

        all_gts[img_id] = gts

        t0 = time.perf_counter()
        p8x = yolo_predict(
            y8x,
            img_path,
            imgsz=args.imgsz,
            conf=args.det_conf,
            iou=args.det_iou,
            max_det=args.max_det
        )
        t_y8x += time.perf_counter() - t0

        t0 = time.perf_counter()
        p8l = yolo_predict(
            y8l,
            img_path,
            imgsz=args.imgsz,
            conf=args.det_conf,
            iou=args.det_iou,
            max_det=args.max_det
        )
        t_y8l += time.perf_counter() - t0

        preds_y8x[img_id] = [
            p for p in p8x
            if p["conf"] >= args.wbf_base_conf
        ]

        preds_y8l[img_id] = [
            p for p in p8l
            if p["conf"] >= args.wbf_base_conf
        ]

        t0 = time.perf_counter()
        fused_all = wbf_fuse(
            [p8x, p8l],
            iou_thr=args.wbf_iou,
            num_models=2
        )
        t_wbf += time.perf_counter() - t0

        preds_wbf[img_id] = [
            p for p in fused_all
            if p["conf"] >= args.wbf_base_conf
        ]

        t0 = time.perf_counter()
        preds_wbf_clip[img_id] = apply_filter_recovery(
            img_path,
            fused_all,
            clip_model,
            clip_preprocess,
            text_features,
            device,
            base_conf=args.wbf_base_conf,
            direct_conf=args.direct_conf,
            recovery_min_conf=args.recovery_min_conf,
            entropy_thr=args.entropy_thr,
            duplicate_iou=args.duplicate_iou
        )
        t_clip += time.perf_counter() - t0

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"[INFO] Processed {idx}/{len(image_paths)} images")

    total_time = time.perf_counter() - start_all

    n_images = len(image_paths)

    fps_y8x = n_images / max(t_y8x, 1e-9)
    fps_y8l = n_images / max(t_y8l, 1e-9)
    fps_wbf = n_images / max(t_y8x + t_y8l + t_wbf, 1e-9)
    fps_wbf_clip = n_images / max(t_y8x + t_y8l + t_wbf + t_clip, 1e-9)

    print("[INFO] Computing metrics...")

    full_knife, full_pistol = count_gts(all_gts)

    results = {
        "dataset": {
            "images": n_images,
            "gt_knife": int(full_knife),
            "gt_handgun_as_pistol": int(full_pistol),
            "gt_total": int(full_knife + full_pistol),
            "class_mapping": {
                "0": "knife",
                "1": "handgun evaluated as pistol"
            }
        },
        "frozen_settings": {
            "detector_conf": args.det_conf,
            "detector_nms_iou": args.det_iou,
            "max_det": args.max_det,
            "wbf_iou": args.wbf_iou,
            "wbf_base_conf": args.wbf_base_conf,
            "direct_conf": args.direct_conf,
            "recovery_min_conf": args.recovery_min_conf,
            "entropy_thr": args.entropy_thr,
            "duplicate_iou": args.duplicate_iou,
            "match_iou": 0.50,
            "clip_model": args.clip_model,
            "clip_pretrained_path": args.clip_pretrained_path,
            "clip_prompts": CLIP_PROMPTS
        },
        "timing_seconds": {
            "yolov8x": round(t_y8x, 4),
            "yolov8l": round(t_y8l, 4),
            "wbf_only": round(t_wbf, 4),
            "clip_filter_recovery": round(t_clip, 4),
            "total_runtime": round(total_time, 4)
        },
        "evaluation": {}
    }

    results["evaluation"]["full_external_dataset"] = {
        "images": n_images,
        "gt_knife": int(full_knife),
        "gt_handgun_as_pistol": int(full_pistol),
        "gt_total": int(full_knife + full_pistol),
        "methods": {
            "YOLOv8x_fixed_0.25": evaluate_method(preds_y8x, all_gts, fps=fps_y8x),
            "YOLOv8l_fixed_0.25": evaluate_method(preds_y8l, all_gts, fps=fps_y8l),
            "YOLOv8x_YOLOv8l_WBF_fixed_0.25": evaluate_method(preds_wbf, all_gts, fps=fps_wbf),
            "YOLOv8x_YOLOv8l_WBF_CLIP_filter_recovery": evaluate_method(preds_wbf_clip, all_gts, fps=fps_wbf_clip)
        }
    }

    detected_splits = sorted(set(split_by_img.values()))

    for split_name in detected_splits:
        if split_name == "full":
            continue

        split_gts = filter_dict_by_split(all_gts, split_by_img, split_name)
        split_y8x = filter_dict_by_split(preds_y8x, split_by_img, split_name)
        split_y8l = filter_dict_by_split(preds_y8l, split_by_img, split_name)
        split_wbf = filter_dict_by_split(preds_wbf, split_by_img, split_name)
        split_wbf_clip = filter_dict_by_split(preds_wbf_clip, split_by_img, split_name)

        split_knife, split_pistol = count_gts(split_gts)

        split_key = f"{split_name}_split"

        results["evaluation"][split_key] = {
            "images": len(split_gts),
            "gt_knife": int(split_knife),
            "gt_handgun_as_pistol": int(split_pistol),
            "gt_total": int(split_knife + split_pistol),
            "methods": {
                "YOLOv8x_fixed_0.25": evaluate_method(split_y8x, split_gts),
                "YOLOv8l_fixed_0.25": evaluate_method(split_y8l, split_gts),
                "YOLOv8x_YOLOv8l_WBF_fixed_0.25": evaluate_method(split_wbf, split_gts),
                "YOLOv8x_YOLOv8l_WBF_CLIP_filter_recovery": evaluate_method(split_wbf_clip, split_gts)
            }
        }

    print("\n========== External Dataset Evaluation ==========")
    print(json.dumps(results, indent=2))

    out_json = out_dir / "external_yolo_mmd_eval_results.json"
    out_csv = out_dir / "external_yolo_mmd_eval_summary.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_csv_summary(results, out_csv)

    print(f"\n[INFO] Saved JSON results to: {out_json}")
    print(f"[INFO] Saved CSV summary to: {out_csv}")

    print("\n[INFO] Expected approximate dataset count from your table:")
    print("       Images = 1710, Knife = 846, Handgun = 1477, Objects = 2329")
    print("       If the printed count is very different, check image-label folder matching.")


if __name__ == "__main__":
    main()