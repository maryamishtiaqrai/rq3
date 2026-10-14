# Fusion-transfer and external evaluation

- `eval_external_yolo_wbf_clip_updated.py`: frozen external-domain evaluation.
- `eval_ccwbf_clip_main_external.py`: CC-WBF-family main/external evaluation.
- `eval_ccwbf_component_ablation_clip.py`: all eight T-W-U component combinations with validation-only selection followed by frozen evaluation.

The paper uses the component analysis to test whether CLIP-FR behavior transfers across related fusion settings.
