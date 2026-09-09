"""Run native PVI-ML population inference for the six-channel Parquet models.

This is an inference-only adapter for ``pvi_ml/scripts/inference_population.py``.
It deliberately keeps PVI-ML's evaluator, result formatter, metric functions,
checkpoint name, and artifact names.  The adaptations are limited to:

* loading the immutable six-channel Parquet cache instead of HDF5;
* selecting CRT or Samba explicitly because the experiment name is not in the
  token order expected by ``ml_session_mapper``;
* rebinding the exact six-channel transform used during training; and
* recovering the original PVI global test order from the frozen split manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

from gcnm_pvi.population_6ch_parquet import PopulationSixChannelParquetDataset
from gcnm_pvi.train_population_6ch_bp import (
    EXPERIMENTS,
    experiment_target,
    mount_pvi_ml,
    six_channel_process_sequence,
)


RESULT_COLUMNS = 50
METADATA_COLUMNS = (
    "sample_id",
    "subject",
    "session",
    "source_name",
    "mask_start",
    "mask_stop",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def original_pvi_test_order(
    dataset: PopulationSixChannelParquetDataset,
) -> tuple[list[int], list[dict]]:
    """Map frozen original-PVI test order to cache-local Parquet row indices."""

    manifest = dataset.manifest
    split_value = manifest.get("split_manifest")
    split_digest = manifest.get("split_manifest_sha256")
    if not split_value or not split_digest:
        raise ValueError("exact PVI cache does not reference its frozen split manifest")
    split_path = Path(split_value).resolve()
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    if _sha256(split_path) != split_digest:
        raise ValueError("frozen PVI split manifest checksum differs from cache manifest")
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if split_manifest.get("schema") not in {
        "pvi-gcnm-population-within-split-v1",
        "pvi-gcnm-population-disjoint-split-v1",
    }:
        raise ValueError("frozen split manifest has the wrong schema")

    cache_by_id: dict[str, tuple[int, dict]] = {}
    test_subset = getattr(dataset, "subsets", {}).get("test")
    if test_subset is not None and hasattr(test_subset, "metadata_rows"):
        cache_rows = test_subset.metadata_rows(list(METADATA_COLUMNS))
    else:
        cache_rows = []
        for relative_path in manifest["files"]["test"]:
            cache_rows.extend(
                pq.read_table(
                    dataset.root / relative_path,
                    columns=list(METADATA_COLUMNS),
                    use_threads=False,
                ).to_pylist()
            )
    for cache_index, row in enumerate(cache_rows):
        sample_id = str(row["sample_id"])
        if sample_id in cache_by_id:
            raise ValueError(f"duplicate test sample_id in Parquet cache: {sample_id}")
        cache_by_id[sample_id] = (cache_index, row)
    cache_index = len(cache_rows)

    expected_count = int(manifest["counts"]["test"])
    if cache_index != expected_count:
        raise ValueError(
            f"test metadata count differs from cache manifest: {cache_index} != {expected_count}"
        )

    local_indices: list[int] = []
    ordered_rows: list[dict] = []
    seen: set[str] = set()
    for identity in split_manifest["identities"]:
        if identity.get("assignment") != "test":
            continue
        sample_id = str(identity["sample_id"])
        try:
            local_index, cache_row = cache_by_id[sample_id]
        except KeyError as exc:
            raise ValueError(
                f"frozen PVI test sample is absent from Parquet cache: {sample_id}"
            ) from exc
        if sample_id in seen:
            raise ValueError(f"duplicate frozen PVI test sample_id: {sample_id}")
        seen.add(sample_id)
        for field in ("subject", "source_name", "mask_start", "mask_stop"):
            if str(identity[field]) != str(cache_row[field]):
                raise ValueError(
                    f"test identity field differs for {sample_id}: "
                    f"{field}={identity[field]!r}/{cache_row[field]!r}"
                )
        local_indices.append(local_index)
        ordered_rows.append(cache_row)

    if len(local_indices) != expected_count or seen != set(cache_by_id):
        raise ValueError(
            "frozen PVI and Parquet test identities differ: "
            f"ordered={len(local_indices)}, cache={len(cache_by_id)}"
        )
    if sorted(local_indices) != list(range(expected_count)):
        raise ValueError("original PVI test order is not a permutation of the cache rows")

    # PVI's population exporter slices one contiguous block per sorted subject.
    subject_blocks = []
    for row in ordered_rows:
        subject = str(row["subject"])
        if not subject_blocks or subject_blocks[-1] != subject:
            subject_blocks.append(subject)
    if subject_blocks != sorted(set(subject_blocks)):
        raise ValueError("original PVI test order is not contiguous and sorted by subject")

    return local_indices, ordered_rows


def make_original_order_loader(
    dataset: PopulationSixChannelParquetDataset,
    local_indices: list[int],
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Reproduce PVI inference's non-stratified, non-shuffled test loader."""

    ordered_test = Subset(dataset.subsets["test"], local_indices)
    worker_options = {
        "num_workers": int(num_workers),
        "pin_memory": True,
    }
    if num_workers > 0:
        worker_options.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )
    return DataLoader(
        ordered_test,
        batch_size=int(batch_size),
        shuffle=False,
        **worker_options,
    )


def _native_metrics(perf_metrics, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    metrics = {
        "bp_accuracy": float(perf_metrics.bp_accuracy(predictions, targets)),
    }
    metrics.update(perf_metrics.metrics_waveform(predictions, targets))
    metrics.update(perf_metrics.metrics_fiducial(predictions, targets))
    return metrics


def _load_results(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    frame = pd.read_csv(path)
    expected = [
        *[f"pred_{index}" for index in range(1, RESULT_COLUMNS + 1)],
        *[f"target_{index}" for index in range(1, RESULT_COLUMNS + 1)],
    ]
    if frame.columns.tolist() != expected:
        raise ValueError(f"existing PVI results have the wrong schema: {path}")
    values = frame.to_numpy(dtype=np.float32)
    if values.shape[1] != 2 * RESULT_COLUMNS or not np.isfinite(values).all():
        raise ValueError(f"existing PVI results are invalid: {path}")
    return torch.from_numpy(values[:, :RESULT_COLUMNS]), torch.from_numpy(
        values[:, RESULT_COLUMNS:]
    )


def assert_metric_parity(
    perf_metrics,
    reference_path: Path,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    tolerance: float,
) -> dict:
    """Require the dedicated pass to reproduce order-independent PVI metrics."""

    reference_predictions, reference_targets = _load_results(reference_path)
    if reference_predictions.shape != predictions.shape:
        raise ValueError(
            "existing and dedicated inference row counts differ: "
            f"{tuple(reference_predictions.shape)} != {tuple(predictions.shape)}"
        )
    reference = _native_metrics(perf_metrics, reference_predictions, reference_targets)
    observed = _native_metrics(perf_metrics, predictions, targets)
    differences = {}
    for key in reference:
        left, right = float(reference[key]), float(observed[key])
        difference = abs(left - right)
        differences[key] = difference
        if not np.isclose(left, right, rtol=tolerance, atol=tolerance, equal_nan=True):
            raise RuntimeError(
                f"dedicated inference metric parity failed for {key}: "
                f"reference={left}, observed={right}, difference={difference}"
            )
    return {
        "reference": reference,
        "observed": observed,
        "max_abs_difference": max(differences.values(), default=0.0),
    }


def assert_checkpoint_accuracy_parity(
    checkpoint: dict,
    observed_metrics: dict,
    *,
    tolerance: float,
) -> dict:
    """Require inference to reproduce the test accuracy saved in a checkpoint."""

    try:
        tracker = checkpoint["tracker"]
        epochs = tracker["epoch"]
        accuracies = tracker["test_accuracy"]
        checkpoint_epoch = int(checkpoint["epoch"])
        tracker_epoch = int(epochs[-1])
        expected = float(accuracies[-1])
        observed = float(observed_metrics["bp_accuracy"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint lacks a valid terminal test accuracy") from exc
    if checkpoint_epoch != tracker_epoch:
        raise ValueError(
            "checkpoint epoch and tracker terminal epoch differ: "
            f"{checkpoint_epoch} != {tracker_epoch}"
        )
    if not np.isfinite(expected) or not np.isfinite(observed):
        raise ValueError("checkpoint or inference test accuracy is non-finite")
    difference = abs(expected - observed)
    if not np.isclose(
        expected,
        observed,
        rtol=tolerance,
        atol=tolerance,
        equal_nan=False,
    ):
        raise RuntimeError(
            "inference does not reproduce the loaded checkpoint test accuracy: "
            f"checkpoint={expected}, observed={observed}, difference={difference}"
        )
    return {
        "checkpoint": expected,
        "observed": observed,
        "absolute_difference": difference,
    }


def _subject_slices(ordered_rows: list[dict]) -> OrderedDict[str, slice]:
    bounds: OrderedDict[str, list[int]] = OrderedDict()
    for index, row in enumerate(ordered_rows):
        subject = str(row["subject"])
        if subject not in bounds:
            bounds[subject] = [index, index + 1]
        else:
            if bounds[subject][1] != index:
                raise ValueError(f"non-contiguous PVI inference rows for {subject}")
            bounds[subject][1] = index + 1
    return OrderedDict(
        (subject, slice(first, stop)) for subject, (first, stop) in bounds.items()
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_inference(args: argparse.Namespace) -> dict:
    mount_pvi_ml(args.pvi_ml_root)
    from src.models import perf_metrics
    from src.models.attn_models import PviCNNTransformer
    from src.models.s4_models import PviSamba
    from src.models.trainer_v3 import ModelEvaluator
    from src.pipeline.data_discovery import ProjectPathManager
    from src.utils.primitives import DEFAULT_TRAIN_DTYPE

    dataset = PopulationSixChannelParquetDataset(
        args.cache_root,
        seed=args.seed,
        require_exact_pvi_schedules=True,
    ).build()
    dataset.set_partition(
        test_size=0.1,
        shuffle=True,
        split_mode=args.split_mode,
        random_state=args.seed,
    )
    local_indices, ordered_rows = original_pvi_test_order(dataset)
    dataset.loaders = {
        "test": make_original_order_loader(
            dataset,
            local_indices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    }

    target = experiment_target(args.experiment, args.split_mode)
    manager = ProjectPathManager(
        branch="main",
        target=target,
        export_root=args.artifact_root,
    )
    checkpoint_path = manager.logdirs["checkpoints"] / args.checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    dataset.load_state_dict(checkpoint["dataset"])
    # Loading checkpoint metadata must not restore a training schedule for
    # inference.  Reapply the original-order loader after validating the state.
    dataset.loaders = {
        "test": make_original_order_loader(
            dataset,
            local_indices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    }

    model_class = PviCNNTransformer if args.experiment == "crt" else PviSamba
    model = model_class(dataset.shapes)
    model._process_sequence = MethodType(six_channel_process_sequence, model)
    evaluator = ModelEvaluator(dataset=dataset, model=model)
    evaluator.checkpoint = checkpoint
    evaluator = evaluator.to(device=args.device, dtype=DEFAULT_TRAIN_DTYPE)
    evaluator.unpack_from_checkpoint("model")
    predictions, targets = evaluator.evaluate_epoch(kw="test")

    expected_shape = (int(dataset.manifest["counts"]["test"]), RESULT_COLUMNS)
    if tuple(predictions.shape) != expected_shape or tuple(targets.shape) != expected_shape:
        raise RuntimeError(
            f"unexpected inference shapes: {tuple(predictions.shape)}/{tuple(targets.shape)}"
        )
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise RuntimeError("dedicated inference produced non-finite values")

    metrics = _native_metrics(perf_metrics, predictions, targets)
    checkpoint_parity = assert_checkpoint_accuracy_parity(
        checkpoint,
        metrics,
        tolerance=args.parity_tolerance,
    )

    existing_results = manager.logdirs["results"] / "dataset_lazy_results.csv"
    parity = None
    if existing_results.is_file() and not args.allow_checkpoint_change:
        parity = assert_metric_parity(
            perf_metrics,
            existing_results,
            predictions,
            targets,
            tolerance=args.parity_tolerance,
        )

    runtime_root = Path(args.runtime_root) if args.runtime_root else None
    if runtime_root is not None:
        runtime_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{target}-inference-", dir=runtime_root) as tmp:
        staging = Path(tmp)
        staging_results = staging / "results"
        staging_statistics = staging / "statistics"
        staging_results.mkdir()
        staging_statistics.mkdir()

        aggregate_path = staging_results / "dataset_lazy_results.csv"
        evaluator.export_results((predictions, targets), aggregate_path)
        for subject, subject_slice in _subject_slices(ordered_rows).items():
            evaluator.export_results(
                (predictions[subject_slice], targets[subject_slice]),
                staging_results / f"{subject}_results.csv",
            )

        # This is exactly PVI-ML's statistics path with compute_wd=False.  The
        # population Wasserstein implementation is quadratic and the original
        # population inference script also leaves that block disabled.
        stats = evaluator.get_stats(compute_wd=False)
        stats_path = staging_statistics / "dataset_lazy_statistics.json"
        evaluator.export_stats(stats, stats_path)

        staged = sorted(staging_results.glob("*.csv")) + [stats_path]
        if not args.no_promote:
            for source in staged:
                destination_root = (
                    manager.logdirs["statistics"]
                    if source.suffix == ".json"
                    else manager.logdirs["results"]
                )
                _atomic_copy(source, destination_root / source.name)

        report = {
            "status": "pass",
            "experiment": args.experiment,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "test_rows": len(ordered_rows),
            "subjects": len(_subject_slices(ordered_rows)),
            "result_columns": 2 * RESULT_COLUMNS,
            "order": "original PVI global test order; non-stratified and non-shuffled",
            "metrics": metrics,
            "checkpoint_accuracy_parity": checkpoint_parity,
            "metric_parity": parity,
            "existing_results_comparison": (
                "intentionally skipped for a promoted checkpoint"
                if existing_results.is_file() and args.allow_checkpoint_change
                else "performed" if existing_results.is_file() else "not available"
            ),
            "promoted": not args.no_promote,
            "results": str((manager.logdirs["results"] / aggregate_path.name).resolve()),
            "statistics": str((manager.logdirs["statistics"] / stats_path.name).resolve()),
        }
        print(json.dumps(report, indent=2))
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--pvi-ml-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--split-mode",
        choices=("within", "disjoint"),
        default="within",
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint-name",
        default="dataset_lazy_checkpoints.pth",
        choices=(
            "dataset_lazy_checkpoints.pth",
            "dataset_lazy_checkpoints_best.pth",
        ),
    )
    parser.add_argument("--parity-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--allow-checkpoint-change",
        action="store_true",
        help=(
            "replace results from an older checkpoint without comparing them to "
            "that obsolete result set; loaded-checkpoint accuracy parity remains required"
        ),
    )
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 32:
        raise ValueError("original PVI population inference uses batch_size=32")
    if args.parity_tolerance < 0:
        raise ValueError("parity tolerance must be non-negative")
    run_inference(args)


if __name__ == "__main__":
    main()
