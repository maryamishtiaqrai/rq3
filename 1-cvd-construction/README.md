# Crop-level Verifier Dataset construction

`create_clip_dataset_from_wbf_pairs.py` constructs the CVD used by the CLIP and YOLO classification verifiers.

- Knife and pistol crops are taken from ground-truth weapon boxes.
- Non-weapon crops are mined from WBF false-positive regions.
- Mining considers all six unordered dual-detector pairs from YOLOv8x, YOLOv8l, YOLO26n, and YOLO26s.
- Hash-identical crop content is retained once within each split/class.
- The manuscript's mining threshold is 0.01 and the maximum is 20 non-weapon crops per image--detector-pair combination.

The script exposes command-line arguments; use the manuscript settings in `../configs/cvd_mining_config.yaml` rather than relying on source-code defaults when reproducing the paper.

A ready-to-edit command using the manuscript settings is provided in `reproduce_cvd_command.txt`.
