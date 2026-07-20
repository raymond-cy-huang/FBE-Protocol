# Mask Benchmark

Generate qualitative comparisons for person foreground masks from BBoxMaskPose, SAM, and SAM2.

```bash
python experiments/mask_benchmark/run.py --limit 3
```

Outputs are written to `results/mask_benchmark/{image_name}/`:

- `input_opencv_box.png`
- `raw_bbmp.png`, `raw_sam.png`, `raw_sam2.png`
- `Mask_bbmp.png`, `Mask_sam.png`, `Mask_sam2.png`
- `comparison.png`
- `boundary_comparison.png`

To inspect SAM/SAM2 candidates before selecting one:

```bash
python experiments/mask_benchmark/show_candidates.py --limit 3
```

To run the first 20 CelebAMask images and write `M_bbox`, `M_sam`, and `M_sam2`:

```bash
python experiments/mask_benchmark/run_celeba20.py
```
