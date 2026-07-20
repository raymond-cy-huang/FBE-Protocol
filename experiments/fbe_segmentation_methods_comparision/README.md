# fbe_segmentation_methods_comparision

Compare BBOX, SAM, and SAM2 segmentation masks on the same image set.

For each input image, the experiment writes:

- `raw_bbox.png`, `raw_sam.png`, `raw_sam2.png`
- `normalized_bbox.png`, `normalized_sam.png`, `normalized_sam2.png`
- `comparison.png`
- `boundary_raw.png`
- `boundary_normalized.png`

It also writes:

- `summary.csv`
- `sam_selection_metadata.csv`
- `pairwise_raw_metrics.csv`
- `pairwise_normalized_metrics.csv`

By default, SAM and SAM2 use the same full-box inverted candidate selection
used by `results/mask_benchmark_full_box_inverted`: prompt with the full image,
select the best multimask candidate, invert it, then normalize that selected
raw mask.

Run from the repo root:

```bash
conda run -n fbe-protocol python experiments/fbe_segmentation_methods_comparision/run.py \
  --input-dir images \
  --output-dir results/fbe_segmentation_methods_comparision
```
