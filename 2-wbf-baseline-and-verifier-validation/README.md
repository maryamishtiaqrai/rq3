# WBF baseline and verifier evaluation

This folder contains the original phase scripts used for the principal validation/test pipeline.

1. `phase1_run_wbf_only_valid.py` generates WBF candidates for all six detector pairs.
2. `phase2_wbf_verifier_valid.py` evaluates CLIP and YOLO-classifier semantic decision rules on validation data.
3. `phase3_apply_frozen_test.py` applies validation-frozen configurations to the MD test split.
4. `phase5_operating_point_diagnostics.py` performs operating-point, matched-recall/matched-precision, threshold-sensitivity, and decision-mode analyses.

Some source files retain the original server path defaults used during the experiments. Override them with command-line arguments on another system.
