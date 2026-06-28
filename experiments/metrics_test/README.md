# Metrics Test

This experiment checks whether every registered `fbe_protocol.metrics` metric is ready in the active environment.

Run from the repository root:

```bash
python experiments/metrics_test/run_metrics_test.py --path-profile metrics_test_path00
```

Inputs:

```text
experiments/metrics_test/input/reference.png
experiments/metrics_test/input/prediction.png
```

Outputs:

```text
experiments/metrics_test/output/metrics_test_result.csv
experiments/metrics_test/output/metrics_setup_status.csv
```

The input and output paths are configured in `configs/global_path.yaml` under
`metrics_test_path00`. You can also override them directly:

```bash
python experiments/metrics_test/run_metrics_test.py \
  --input-dir /path/to/input \
  --output-dir /path/to/output \
  --reference-image reference.png \
  --prediction-image prediction.png
```
