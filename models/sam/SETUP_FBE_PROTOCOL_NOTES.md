# SAM v1 setup notes for fbe-protocol

Date: 2026-07-04

Target environment:

- Conda env: `fbe-protocol`
- Python: `3.10.20`
- PyTorch: `2.4.0`
- TorchVision: `0.19.0`
- CUDA available: yes
- GPU: `NVIDIA GeForce RTX 3070`

## What was done

1. Checked the SAM v1 repository at `/home/rz/raymond_project/sam`.
2. Confirmed the `fbe-protocol` env already had the required base runtime:
   `torch`, `torchvision`, `opencv-python`, `pycocotools`, `onnx`, and `onnxruntime`.
3. Installed SAM v1 into the `fbe-protocol` env as an editable local package:

   ```bash
   cd /home/rz/raymond_project/sam
   /home/rz/miniconda3/envs/fbe-protocol/bin/python -m pip install -e .
   ```

4. Downloaded the smallest SAM v1 checkpoint for minimal testing:

   ```bash
   mkdir -p /home/rz/raymond_project/sam/checkpoints
   cd /home/rz/raymond_project/sam/checkpoints
   wget -O sam_vit_b_01ec64.pth \
     https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
   ```

   Downloaded file:

   ```text
   /home/rz/raymond_project/sam/checkpoints/sam_vit_b_01ec64.pth
   size: 375042383 bytes, about 358M
   model_type: vit_b
   ```

5. Ran a SAM repo smoke test with `notebooks/images/truck.jpg`.
6. Ran a porting-oriented smoke test from `/home/rz/raymond_project/FBE-Protocol`
   using `images/15881.png`, without manually adding the SAM repo to `PYTHONPATH`.

## Minimal viable tests

### SAM repo test

Command summary:

```bash
cd /home/rz/raymond_project/sam
/home/rz/miniconda3/envs/fbe-protocol/bin/python -c "... SamPredictor vit_b on notebooks/images/truck.jpg ..."
```

Result:

```text
device cuda
image (1200, 1800, 3)
masks (3, 1200, 1800) bool
scores [0.9005862474441528, 0.8546345829963684, 0.8763855695724487]
best_mask_pixels 628528
output results/sam_smoke/truck_center_mask.png
```

### FBE-Protocol import and inference test

Command summary:

```bash
cd /home/rz/raymond_project/FBE-Protocol
/home/rz/miniconda3/envs/fbe-protocol/bin/python -c "... import segment_anything; SamPredictor vit_b on images/15881.png ..."
```

Result:

```text
segment_anything_import_ok
device cuda
image (512, 512, 3)
scores [0.9826790690422058, 0.9911945462226868, 0.9656943082809448]
best_mask_pixels 109215
output /tmp/fbe_protocol_sam_v1_smoke/15881_sam_vit_b_mask.png
```

## Porting notes

- The `segment_anything` package is now importable from the `fbe-protocol` env.
- FBE code can use:

  ```python
  from segment_anything import SamPredictor, sam_model_registry
  ```

- Recommended initial checkpoint for integration tests:

  ```text
  /home/rz/raymond_project/sam/checkpoints/sam_vit_b_01ec64.pth
  ```

- Existing larger SAM v1 checkpoint found elsewhere:

  ```text
  /home/rz/raymond_project/Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth
  ```

  This was not used for the minimal test because `vit_b` is lighter and enough
  to verify the environment and API.

## Warnings observed

- `onnxruntime` printed a GPU device discovery warning, but SAM v1 prompt
  inference does not depend on ONNX Runtime and the tests passed.
- PyTorch printed a `torch.load(weights_only=False)` future warning from
  `segment_anything/build_sam.py`. This is upstream behavior and did not block
  loading the checkpoint.
