#!/usr/bin/env python3
"""Build auditable tables for the coarse and projected-fine GCNM versions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


RINGS = tuple(f"US{size:03d}" for size in range(60, 131, 5))
TRAINING_VERSIONS = (
    {
        "key": "original_coarse",
        "label": "Original: coarse training",
        "experiment": "coordinate_direct",
        "physics": "F_c(sigma_c), J_c",
    },
    {
        "key": "retrained_projected_fine",
        "label": "Retrained: projected-fine training",
        "experiment": "coordinate_direct_projected_fine",
        "physics": "F_f(P sigma_c), J_f P",
    },
)
EVALUATIONS = (
    {
        "key": "original_coarse",
        "label": "Original",
        "training_physics": "F_c(sigma_c), J_c",
        "inference_physics": "F_c(sigma_c), J_c",
        "report": "coordinate_direct/exact_nonlinear_test_coarse_20260816/report.json",
        "predictions": "coordinate_direct/exact_nonlinear_test_coarse_20260816/predictions.npz",
    },
    {
        "key": "original_weights_projected_fine",
        "label": "Rewired, fixed weights",
        "training_physics": "F_c(sigma_c), J_c",
        "inference_physics": "F_f(P sigma_c), J_f P",
        "report": (
            "coordinate_direct/"
            "exact_nonlinear_test_fixed_weights_projected_fine_20260816/report.json"
        ),
        "predictions": (
            "coordinate_direct/"
            "exact_nonlinear_test_fixed_weights_projected_fine_20260816/predictions.npz"
        ),
    },
    {
        "key": "retrained_projected_fine",
        "label": "Retrained projected-fine",
        "training_physics": "F_f(P sigma_c), J_f P",
        "inference_physics": "F_f(P sigma_c), J_f P",
        "report": "coordinate_direct_projected_fine/exact_nonlinear_test/report.json",
        "predictions": "coordinate_direct_projected_fine/exact_nonlinear_test/predictions.npz",
    },
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _training_report(repo: Path, ring: str, experiment: str) -> Path:
    if ring == "US120":
        root = repo / "data/differential_US120_1000beats_results_v1"
    else:
        root = repo / "data/differential_main_b045_1000beats_results_v1" / ring
    return root / experiment / f"{experiment}_training_report.json"


def _scalar_settings(report: dict) -> dict:
    excluded = {"history", "physics_stages", "artifact_hashes"}
    return {
        key: value
        for key, value in report.items()
        if key not in excluded and isinstance(value, (str, int, float, bool, type(None)))
    }


def _training_tables(repo: Path) -> dict[str, list[dict]]:
    runs: list[dict] = []
    epochs: list[dict] = []
    selected: list[dict] = []
    physics: list[dict] = []
    hashes: list[dict] = []
    dataset_hashes: dict[tuple[str, str], tuple[str, str]] = {}

    for version in TRAINING_VERSIONS:
        for ring in RINGS:
            path = _training_report(repo, ring, version["experiment"])
            if not path.is_file():
                raise FileNotFoundError(path)
            report = _read_json(path)
            base = {
                "ring": ring,
                "training_version": version["key"],
                "training_label": version["label"],
                "training_physics": version["physics"],
                "report_path": str(path.resolve()),
            }
            runs.append({**base, **_scalar_settings(report)})
            artifact_hashes = report["artifact_hashes"]
            hashes.append({**base, **artifact_hashes})
            dataset_pair = (
                artifact_hashes["train_dataset_sha256"],
                artifact_hashes["validation_dataset_sha256"],
            )
            previous = dataset_hashes.setdefault((ring, "shared"), dataset_pair)
            if previous != dataset_pair:
                raise ValueError(f"training datasets differ between versions for {ring}")

            if len(report["history"]) != 2 or len(report["physics_stages"]) != 2:
                raise ValueError(f"expected two stages in {path}")
            for stage_index, history in enumerate(report["history"], start=1):
                if not history:
                    raise ValueError(f"empty stage {stage_index} history in {path}")
                for entry in history:
                    epochs.append({**base, "stage": stage_index, **entry})
                best = min(history, key=lambda row: row["composite_score"])
                selected.append(
                    {
                        **base,
                        "stage": stage_index,
                        "epochs_run": len(history),
                        "selected_epoch": best["epoch"],
                        "selected_train_loss": best["train_loss"],
                        "selected_validation_loss": best["validation_loss"],
                        "selected_validation_nrmse": best["nrmse"],
                        "selected_validation_background_rms_normalized": best[
                            "background_rms_normalized"
                        ],
                        "selected_validation_dice_global": best["dice_global"],
                        "selected_validation_composite_score": best["composite_score"],
                    }
                )
                stage_physics = report["physics_stages"][stage_index - 1]
                if int(stage_physics["stage"]) != stage_index:
                    raise ValueError(f"stage mismatch in {path}")
                for split in ("train", "validation"):
                    physics.append(
                        {
                            **base,
                            "stage": stage_index,
                            "split": split,
                            **stage_physics[split],
                        }
                    )
    return {
        "training_run_settings.csv": runs,
        "training_epoch_metrics_all.csv": epochs,
        "training_selected_checkpoints.csv": selected,
        "training_physics_diagnostics.csv": physics,
        "training_artifact_hashes.csv": hashes,
    }


def _aggregate_selected(rows: list[dict]) -> list[dict]:
    metrics = (
        "selected_train_loss",
        "selected_validation_loss",
        "selected_validation_nrmse",
        "selected_validation_background_rms_normalized",
        "selected_validation_dice_global",
        "selected_validation_composite_score",
        "selected_epoch",
        "epochs_run",
    )
    output: list[dict] = []
    for version in TRAINING_VERSIONS:
        for stage in (1, 2):
            group = [
                row
                for row in rows
                if row["training_version"] == version["key"] and row["stage"] == stage
            ]
            if len(group) != len(RINGS):
                raise ValueError(f"incomplete aggregate group {version['key']} stage {stage}")
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
                output.append(
                    {
                        "training_version": version["key"],
                        "training_label": version["label"],
                        "stage": stage,
                        "metric": metric,
                        "rings": len(values),
                        "mean": float(np.mean(values)),
                        "standard_deviation": float(np.std(values, ddof=1)),
                        "median": float(np.median(values)),
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                    }
                )
    return output


def _three_version_tables(repo: Path) -> tuple[list[dict], dict]:
    root = repo / "data/differential_US120_1000beats_results_v1"
    rows: list[dict] = []
    arrays = []
    reports = []
    for evaluation in EVALUATIONS:
        report_path = root / evaluation["report"]
        predictions_path = root / evaluation["predictions"]
        report = _read_json(report_path)
        archive = np.load(predictions_path)
        reports.append(report)
        arrays.append(
            {
                "truth": np.asarray(archive["truth"]),
                "old_newton": np.asarray(archive["old_newton"]),
            }
        )
        if int(report["samples"]) != 5000 or len(report["faithful_gcnm"]) != 2:
            raise ValueError(f"unexpected evaluation contract in {report_path}")
        for stage in report["faithful_gcnm"]:
            rows.append(
                {
                    "evaluation_version": evaluation["key"],
                    "evaluation_label": evaluation["label"],
                    "training_physics": evaluation["training_physics"],
                    "inference_physics": evaluation["inference_physics"],
                    "ring": "US120",
                    "test_samples": report["samples"],
                    "stage": stage["stage"],
                    **stage["metrics"],
                    **{f"physics_{key}": value for key, value in stage["physics"].items()},
                    "newton_relative_classification": stage[
                        "relative_to_saved_one_step_newton"
                    ]["classification"],
                    "newton_relative_improvements": stage[
                        "relative_to_saved_one_step_newton"
                    ]["improvements"],
                    "report_path": str(report_path.resolve()),
                    "predictions_path": str(predictions_path.resolve()),
                }
            )
    identical_truth = all(
        np.array_equal(arrays[0]["truth"], item["truth"], equal_nan=True)
        for item in arrays[1:]
    )
    identical_newton = all(
        np.array_equal(arrays[0]["old_newton"], item["old_newton"], equal_nan=True)
        for item in arrays[1:]
    )
    if not identical_truth or not identical_newton:
        raise ValueError("the three evaluations do not share identical test references")
    reference = {
        "ring": "US120",
        "test_samples": 5000,
        "truth_arrays_identical": identical_truth,
        "saved_newton_arrays_identical": identical_newton,
        "truth_shape": list(arrays[0]["truth"].shape),
        "saved_newton_metrics": reports[0]["saved_newton"],
        "comparison_scope": (
            "Controlled exact nonlinear test comparison on US120. The first and third "
            "versions were independently trained. The middle version reuses the first "
            "version's weights and changes only the stage-physics wiring at inference."
        ),
    }
    return rows, reference


def _deltas(rows: list[dict]) -> list[dict]:
    metrics = (
        "element_nrmse",
        "image_correlation",
        "localization_dice_at_true_volume",
        "support_sign_accuracy",
        "background_rms",
        "prediction_to_truth_rms",
        "frame_difference_rms_ratio",
        "physics_post_gcn_voltage_residual_rms_mean",
    )
    output = []
    for stage in (1, 2):
        stage_rows = {row["evaluation_version"]: row for row in rows if row["stage"] == stage}
        original = stage_rows["original_coarse"]
        for key in ("original_weights_projected_fine", "retrained_projected_fine"):
            candidate = stage_rows[key]
            for metric in metrics:
                baseline = float(original[metric])
                value = float(candidate[metric])
                output.append(
                    {
                        "stage": stage,
                        "comparison": f"{key}_minus_original_coarse",
                        "metric": metric,
                        "original_coarse": baseline,
                        "candidate": value,
                        "absolute_difference": value - baseline,
                        "relative_difference_percent": (
                            100.0 * (value - baseline) / baseline if baseline != 0 else np.nan
                        ),
                    }
                )
    return output


def _latex_table(rows: list[dict]) -> str:
    labels = {entry["key"]: entry["label"] for entry in EVALUATIONS}
    by_key = {(row["evaluation_version"], row["stage"]): row for row in rows}
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Comparison of the three GCNM implementations on the same 5,000-sample US120 exact nonlinear test set. Element NRMSE and background RMS are lower-is-better, while image correlation and localization Dice are higher-is-better. The fixed-weight version changes only the physics used during inference and was not retrained.}",
        r"    \label{tab:gcnm-three-version-comparison}",
        r"    \small",
        r"    \begin{tabular}{llrrrr}",
        r"        \hline",
        r"        Version & Stage & Element NRMSE & Image correlation & Localization Dice & Background RMS (S/m) \\",
        r"        \hline",
    ]
    for evaluation in EVALUATIONS:
        for stage in (1, 2):
            row = by_key[(evaluation["key"], stage)]
            label = labels[evaluation["key"]] if stage == 1 else ""
            lines.append(
                "        "
                + f"{label} & {stage} & {row['element_nrmse']:.3f} & "
                + f"{row['image_correlation']:.3f} & "
                + f"{row['localization_dice_at_true_volume']:.3f} & "
                + f"{row['background_rms']:.3e} \\\\"
            )
    lines.extend(
        [
            r"        \hline",
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _training_latex_table(rows: list[dict]) -> str:
    by_key = {
        (row["training_version"], int(row["stage"]), row["metric"]): row
        for row in rows
    }
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Selected-checkpoint validation metrics for the two trained GCNM versions across the 15 ring geometries. Values are mean $\pm$ standard deviation across rings. NRMSE, background RMS, and the composite selection score are lower-is-better, while Dice is higher-is-better.}",
        r"    \label{tab:gcnm-training-version-summary}",
        r"    \small",
        r"    \begin{tabular}{llrrrr}",
        r"        \hline",
        r"        Training version & Stage & NRMSE & Background RMS & Dice & Selection score \\",
        r"        \hline",
    ]
    for version in TRAINING_VERSIONS:
        for stage in (1, 2):
            values = {}
            for short, metric in (
                ("nrmse", "selected_validation_nrmse"),
                ("background", "selected_validation_background_rms_normalized"),
                ("dice", "selected_validation_dice_global"),
                ("score", "selected_validation_composite_score"),
            ):
                row = by_key[(version["key"], stage, metric)]
                values[short] = f"{float(row['mean']):.3f} $\\pm$ {float(row['standard_deviation']):.3f}"
            label = version["label"] if stage == 1 else ""
            lines.append(
                f"        {label} & {stage} & {values['nrmse']} & "
                f"{values['background']} & {values['dice']} & {values['score']} \\\\"
            )
    lines.extend(
        [
            r"        \hline",
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    training = _training_tables(repo)
    for filename, rows in training.items():
        _write_csv(output / filename, rows)
    selected = training["training_selected_checkpoints.csv"]
    training_summary = _aggregate_selected(selected)
    _write_csv(output / "training_cross_ring_summary.csv", training_summary)
    (output / "training_cross_ring_summary_table.tex").write_text(
        _training_latex_table(training_summary), encoding="utf-8"
    )

    comparison, reference = _three_version_tables(repo)
    _write_csv(output / "three_version_exact_test_metrics_all.csv", comparison)
    _write_csv(output / "three_version_differences_from_original.csv", _deltas(comparison))
    (output / "three_version_exact_test_contract.json").write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "three_version_comparison_table.tex").write_text(
        _latex_table(comparison), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# GCNM version metrics\n\n"
        "The two training versions are the original coarse-physics GCNM and the "
        "retrained projected-fine GCNM. `training_epoch_metrics_all.csv` contains every "
        "reported epoch metric for all 15 rings and both stages. The selected-checkpoint, "
        "physics-diagnostic, run-setting, artifact-hash, and cross-ring summary tables are "
        "provided separately.\n\n"
        "The three-version accuracy comparison uses the identical 5,000-sample US120 exact "
        "nonlinear test arrays. `original_weights_projected_fine` is not a third trained "
        "architecture: it applies the original weights after changing only inference physics "
        "from `F_c, J_c` to `F_f(P sigma_c), J_f P`. The third version was retrained with "
        "that projected-fine physics. Experimental participant images have no conductivity "
        "ground truth and are therefore not used to claim reconstruction accuracy.\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "pass",
        "rings": list(RINGS),
        "training_versions": [entry["key"] for entry in TRAINING_VERSIONS],
        "evaluation_versions": [entry["key"] for entry in EVALUATIONS],
        "training_report_count": len(RINGS) * len(TRAINING_VERSIONS),
        "training_stage_count": len(selected),
        "training_epoch_rows": len(training["training_epoch_metrics_all.csv"]),
        "three_version_metric_rows": len(comparison),
        "exact_test_contract": reference,
        "files": sorted(path.name for path in output.iterdir()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
