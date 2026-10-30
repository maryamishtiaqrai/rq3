# CLIP Guided Filter Recovery for Weapon Detection Using Dual YOLO Box Fusion
*(Source code, selected experimental outputs, and reproducibility resources for the associated manuscript)*

**Journal:** Advanced Intelligent Systems  
**Paper Status:** Submitted

This repository contains the experimental code and selected result files used for the manuscript:

> **CLIP Guided Filter Recovery for Weapon Detection Using Dual YOLO Box Fusion**  
> Authors: Muhammad Ishtiaq, Mingchu Li, Sanaullah Manzoor, Talha Mahboob Alam, Junaid Abdullah Mansoor, Maryam Gulzar

The main framework uses YOLOv8x and YOLOv8l for localization, class-wise Weighted Boxes Fusion (WBF) for spatial fusion, and a fine-tuned OpenCLIP ViT-B/32 model as a crop-level semantic verifier. The final CLIP-FR decision rule selectively filters intermediate-confidence candidates and recovers semantically supported low-confidence candidates.

---

## Repository Layout

```text
1-cvd-construction/
├── create_clip_dataset_from_wbf_pairs.py
└── README.md

2-wbf-baseline-and-verifier-validation/
├── phase1_run_wbf_only_valid.py
├── phase2_wbf_verifier_valid.py
├── phase3_apply_frozen_test.py
├── phase5_operating_point_diagnostics.py
└── README.md

3-single-yolo-baseline/
└── phase6_single_yolo_all_detectors.py

4-ccwbf-transfer-and-external-evaluation/
├── eval_external_yolo_wbf_clip_updated.py
├── eval_ccwbf_clip_main_external.py
├── eval_ccwbf_component_ablation_clip.py
└── README.md

5-hard-negative-ablation/
├── phase8_build_d5_hn_dataset.py
├── phase8_finetune_hn_multi_yolo.py
├── phase9_hn_detector_wbf_evaluation.py
├── phase9_hn_detector_wbf_evaluation_MAP_FINAL.py
└── phase10_hn_wbf_verifier_final_analysis.py

6-qualitative-analysis/
└── render_manual_selected_cases.py

7-results/
├── phase5_selected_pair_test_results.csv
├── phase5_test_wbf_threshold_curve.csv
├── phase5_test_matched_recall_vs_wbf.csv
├── phase5_test_matched_precision_vs_wbf.csv
├── ccwbf_component_ablation_summary.csv
├── ccwbf_main_external_summary.csv
└── phase10_hn_selected_pair_test_results.csv

configs/
├── clip_fr_config.yaml
├── clip_verifier_config.yaml
└── cvd_mining_config.yaml

scripts/
└── update_repo_metadata.py

8-Dataset.txt
CITATION.cff
USAGE_NOTICE.txt
repo_config.json
requirements.txt
```

---

## Main Experimental Pipeline

1. **Detector pool**: YOLOv8x, YOLOv8l, YOLO26n, and YOLO26s.
2. **Main detector pair**: YOLOv8x + YOLOv8l.
3. **Fusion**: class-wise WBF with IoU 0.55, equal detector weights, average confidence, and minimum contributor rescaling.
4. **Verifier**: OpenCLIP ViT-B/32 fine-tuned on the MD-derived Crop-level Verifier Dataset (CVD).
5. **Final semantic rule**: verifier class agreement plus normalized entropy threshold.
6. **Final CLIP-FR thresholds**: baseline 0.25, direct acceptance 0.50, recovery minimum 0.15, entropy threshold 0.20.
7. **Recovery duplicate check**: same-class IoU threshold 0.50 against already retained detections.

The final values are also summarized in `configs/clip_fr_config.yaml`.

---

## Crop-level Verifier Dataset (CVD)

The CVD is derived from the Main Dataset (MD):

- **Knife/pistol**: ground-truth weapon crops with contextual padding.
- **Non-weapon**: target-domain false-positive regions mined from WBF output.
- Non-weapon mining uses all six unordered dual-detector pairs from the four-model detector pool.
- Crop content with an identical normalized hash is retained once within each split/class.
- The paper uses a mining threshold of 0.01 and at most 20 non-weapon crops per image--detector-pair combination.

Final class counts:

| Split | Knife | Pistol | Non-weapon | Total |
|---|---:|---:|---:|---:|
| Train | 3,013 | 3,013 | 3,013 | 9,039 |
| Validation | 1,448 | 1,216 | 1,124 | 3,788 |
| Test | 1,447 | 1,217 | 1,052 | 3,716 |
| **Total** | **5,908** | **5,446** | **5,189** | **16,543** |

See `1-cvd-construction/` and `configs/cvd_mining_config.yaml`.

---

## Main Reported Result

On the MD test split, the selected YOLOv8x+YOLOv8l fixed-WBF baseline produced 76 FP and 156 FN. The final CLIP-FR configuration produced 73 FP and 147 FN, corresponding to relative reductions of 3.95% and 5.77%, respectively, with F1 increasing from 0.9558 to 0.9582.

The repository also includes selected files for threshold sensitivity, matched operating-point comparisons, CC-WBF-family component transfer, external evaluation, and hard-negative detector-training ablation.

---

## Environment and Dependencies

The reported experiments used Python 3.10.6, PyTorch 2.0.1 + cu118, Ultralytics 8.4.96, OpenCLIP 3.2.0, and an NVIDIA Tesla V100-PCIE-32GB GPU.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Model checkpoints and third-party datasets are not bundled in this compact repository archive. Supply their paths through the command-line interfaces of the experimental scripts.

---

## Reproducing the Main Workflow

1. Download and prepare the image-level weapon dataset described in `8-Dataset.txt`.
2. Train or provide the four detector checkpoints used by the experimental scripts.
3. Run `1-cvd-construction/create_clip_dataset_from_wbf_pairs.py` using the paper settings in `configs/cvd_mining_config.yaml`.
4. Provide the trained CLIP verifier and supervised YOLO classification verifier checkpoints.
5. Run the WBF baseline and validation scripts in `2-wbf-baseline-and-verifier-validation/`.
6. Freeze the validation-selected configuration and evaluate it on the MD test split.
7. Use folders 4--6 for the fusion-transfer, external-domain, hard-negative, and qualitative analyses.

The original research scripts preserve several server-default paths from the experiments. Override these paths through their command-line arguments when running on a different machine.

---

## Paper Metadata from One JSON File

Edit only `repo_config.json`:

```json
{
  "journal_name": "Advanced Intelligent Systems",
  "paper_status": "Submitted"
}
```

Then run:

```bash
python scripts/update_repo_metadata.py
```

The script updates the journal/status lines in the repository metadata files. For example, after acceptance you can change only `paper_status` to `Accepted`, run the script, and commit the updated files.

---

## Citation

Until publication metadata/DOI are available, cite the repository/manuscript using the information in `CITATION.cff`.

```bibtex
@article{ishtiaq2026clipfr,
  title={CLIP Guided Filter Recovery for Weapon Detection Using Dual YOLO Box Fusion},
  author={Ishtiaq, Muhammad and Li, Mingchu and Manzoor, Sanaullah and Alam, Talha Mahboob and Mansoor, Junaid Abdullah and Gulzar, Maryam},
  journal={Advanced Intelligent Systems},
  year={2026},
  note={Paper status: Submitted}
}
```

---

## Usage Policy

Copyright © 2026 Muhammad Ishtiaq and co-authors.  
See `USAGE_NOTICE.txt` for the repository usage notice and dataset-redistribution note.

---

## Contact

**First/corresponding author:** Muhammad Ishtiaq  
**E-mail:** ishtiaqrai8@gmail.com
