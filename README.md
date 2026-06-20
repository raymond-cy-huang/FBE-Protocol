# FBE-Protocol : Foreground-Background Entanglement and Background Fidelity in Generative Image Inversion.



This repository provides mask extraction utilities built upon
[MiraPurkrabek/BBoxMaskPose](https://github.com/MiraPurkrabek/BBoxMaskPose),
which is used as the raw mask generation backend, and the normalization
pipeline in `fbe_protocol/mask`.

## Repository Layout

```text
FBE-Protocol/
+-- configs/
|   +-- global_path.yaml          # Input/output path profiles for scripts
+-- fbe_protocol/
|   +-- mask/                     # Binary mask extraction package
+-- images/                       # Default local input images
+-- models/
|   +-- BBoxMaskPose/             # Raw mask generator
+-- results/                      # Default generated outputs
+-- scripts/
|   +-- fbe_extract_mask.py       # Single-image mask extraction
|   +-- fbe_extract_multi_masks.py # Folder/batch mask extraction
|   +-- task_trigger.sh           # Interactive script launcher
+-- environment.yml
+-- setup.sh
```

## Setup

Run setup from the repository root:

```bash
bash setup.sh
```

`setup.sh` performs the following steps:

1. Creates or updates the conda environment from `environment.yml`.
2. Installs OpenMMLab dependencies with `mim`:
   `mmengine`, `mmcv==2.1.0`, `mmdet==3.3.0`, and `mmpretrain==1.2.0`.
3. Installs BBoxMaskPose runtime dependencies.
4. Downloads SAM checkpoints into:
   `models/BBoxMaskPose/models/SAM/`

The default environment name is `fbe-protocol`.

```bash
conda activate fbe-protocol
```

To use a different environment name:

```bash
FBE_CONDA_ENV=my-env-name bash setup.sh
conda activate my-env-name
```

To retry network-dependent setup steps more times:

```bash
FBE_SETUP_RETRIES=5 bash setup.sh
```

## Model Checkpoints

The setup script downloads these SAM checkpoints when any of them are missing:

```text
models/BBoxMaskPose/models/SAM/sam2.1_hiera_tiny.pt
models/BBoxMaskPose/models/SAM/sam2.1_hiera_small.pt
models/BBoxMaskPose/models/SAM/sam2.1_hiera_base_plus.pt
models/BBoxMaskPose/models/SAM/sam2.1_hiera_large.pt
```

If the checkpoints already exist, `setup.sh` skips the download.

## Run Scripts

### Single Image

Run BBoxMaskPose on one image, normalize the raw mask, and write the output:

```bash
python scripts/fbe_extract_mask.py images/15881.png
```

Default output:

```text
results/fbe_extract_mask/15881/15881_mask.png
results/fbe_extract_mask/15881/15881_combine.png
```

Useful options:

```bash
python scripts/fbe_extract_mask.py images/15881.png \
  --output-dir results \
  --output-layout task \
  --output-mode full
```

`--output-layout task` writes task/image subfolders.
`--output-layout direct` writes directly under the output folder.

`--output-mode full` writes both mask and combine image.
`--output-mode mask_only` writes only the binary mask.

If you already have a raw mask and want to skip BBoxMaskPose:

```bash
python scripts/fbe_extract_mask.py images/15881.png \
  --mask path/to/raw_mask.jpg
```

### Multiple Images

Use the configured folder profile from `configs/global_path.yaml`:

```bash
python scripts/fbe_extract_multi_masks.py --path-profile fbe_extract_multi_masks_path00
```

Override paths from the command line:

```bash
python scripts/fbe_extract_multi_masks.py \
  --input-dir images \
  --output-dir results \
  --output-layout task \
  --output-mode full
```

### Interactive Launcher

Use the interactive task selector:

```bash
bash scripts/task_trigger.sh
```

The launcher lists available scripts under `scripts/`, then lists path profiles
from `configs/global_path.yaml`.

## Update Paths and Configs

Edit:

```text
configs/global_path.yaml
```

The default profile is:

```yaml
fbe_extract_multi_masks_path00:
  input_dir: images
  output_dir: results
  output_layout: task
  output_mode: full
```

Relative paths are resolved from the repository root. Absolute paths are also
supported.

For a new experiment, add another profile:

```yaml
fbe_extract_multi_masks_path02:
  input_dir: /absolute/path/to/input_images
  output_dir: /absolute/path/to/output_masks
  output_layout: direct
  output_mode: mask_only
```

Then run:

```bash
python scripts/fbe_extract_multi_masks.py --path-profile fbe_extract_multi_masks_path02
```

## Output Naming

For `output_layout: task` and `output_mode: full`:

```text
results/fbe_extract_multi_masks/{image_name}/{image_name}_mask.png
results/fbe_extract_multi_masks/{image_name}/{image_name}_combine.png
```

For `output_layout: direct` and `output_mode: mask_only`:

```text
{output_dir}/{image_name}_mask.png
```

## Mask Normalization

The package API is:

```python
from fbe_protocol.mask import MaskConfig, extract

result = extract(image=image_bgr, mask=raw_mask, config=MaskConfig())
binary_mask = result.mask
```

The output mask is always `uint8` with values `{0, 255}`.
