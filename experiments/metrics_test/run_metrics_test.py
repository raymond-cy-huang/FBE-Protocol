#!/usr/bin/env python3
"""Run a readiness smoke test for all registered FBE metrics."""

from __future__ import annotations

import csv
import argparse
import sys
import traceback
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
CONFIG_PATH = REPO_ROOT / "configs/global_path.yaml"
DEFAULTS = {
    "input_dir": EXPERIMENT_ROOT / "input",
    "output_dir": EXPERIMENT_ROOT / "output",
    "reference_image": Path("reference.png"),
    "prediction_image": Path("prediction.png"),
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fbe_protocol.metrics import MetricsEnvChecker, MetricsTaskerBuilder, list_metrics


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_input_path(path: Path, input_dir: Path) -> Path:
    return path if path.is_absolute() else input_dir / path


def read_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read configs/global_path.yaml. Run `bash setup.sh`.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


def apply_profile(args: argparse.Namespace) -> argparse.Namespace:
    if not args.path_profile:
        return args

    config = read_config(CONFIG_PATH)
    section = config.get(args.path_profile, {})
    if not isinstance(section, dict):
        raise ValueError(f"Config profile must be a YAML object: {args.path_profile}")

    def set_if_present(attr: str, key: str, caster=lambda value: value) -> None:
        if key not in section or section[key] is None:
            return
        if getattr(args, attr) == DEFAULTS[attr]:
            setattr(args, attr, caster(section[key]))

    set_if_present("input_dir", "input_dir", Path)
    set_if_present("output_dir", "output_dir", Path)
    set_if_present("reference_image", "reference_image", Path)
    set_if_present("prediction_image", "prediction_image", Path)
    return args


def build_setup_rows(metric_names: list[str], checker: MetricsEnvChecker) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in metric_names:
        report = checker.check(metric)
        rows.append(
            {
                "metric": report.metric,
                "ready": report.available,
                "missing_modules": ", ".join(report.missing_modules),
            }
        )
    return rows


def run_metric(metric: str, checker: MetricsEnvChecker, reference_image: Path, prediction_image: Path) -> dict[str, object]:
    report = checker.check(metric)
    row: dict[str, object] = {
        "metric": report.metric,
        "dependency_ready": report.available,
        "run_status": "skipped",
        "score": "",
        "error": "",
    }
    if not report.available:
        row["error"] = f"missing modules: {', '.join(report.missing_modules)}"
        return row

    try:
        tasker = MetricsTaskerBuilder(check_environment=True).add(metric).build()
        scores = tasker.evaluate_pair(reference_image, prediction_image)
        row["run_status"] = "passed"
        row["score"] = scores[report.metric]
    except Exception as exc:
        row["run_status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-profile", default=None, help="Profile name in configs/global_path.yaml.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULTS["input_dir"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULTS["output_dir"])
    parser.add_argument("--reference-image", type=Path, default=DEFAULTS["reference_image"])
    parser.add_argument("--prediction-image", type=Path, default=DEFAULTS["prediction_image"])
    return apply_profile(parser.parse_args())


def main() -> int:
    args = parse_args()
    input_dir = resolve_repo_path(args.input_dir)
    output_dir = resolve_repo_path(args.output_dir)
    reference_image = resolve_input_path(args.reference_image, input_dir)
    prediction_image = resolve_input_path(args.prediction_image, input_dir)
    result_csv = output_dir / "metrics_test_result.csv"
    setup_status_csv = output_dir / "metrics_setup_status.csv"

    if not reference_image.exists() or not prediction_image.exists():
        raise FileNotFoundError(
            f"Expected test inputs at {reference_image} and {prediction_image}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checker = MetricsEnvChecker()
    metric_names = list_metrics()

    setup_rows = build_setup_rows(metric_names, checker)
    write_csv(
        setup_status_csv,
        setup_rows,
        fieldnames=["metric", "ready", "missing_modules"],
    )

    result_rows = [run_metric(metric, checker, reference_image, prediction_image) for metric in metric_names]
    write_csv(
        result_csv,
        result_rows,
        fieldnames=["metric", "dependency_ready", "run_status", "score", "error", "traceback"],
    )

    passed = sum(row["run_status"] == "passed" for row in result_rows)
    failed = sum(row["run_status"] == "failed" for row in result_rows)
    skipped = sum(row["run_status"] == "skipped" for row in result_rows)
    print(f"input: {input_dir}")
    print(f"reference: {reference_image}")
    print(f"prediction: {prediction_image}")
    print(f"output: {output_dir}")
    print(f"metrics tested: {len(result_rows)}")
    print(f"passed: {passed}, failed: {failed}, skipped: {skipped}")
    print(f"result: {result_csv}")
    print(f"setup status: {setup_status_csv}")
    return 1 if failed or skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
