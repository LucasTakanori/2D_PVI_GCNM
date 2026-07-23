"""Run one original-protocol pvi_ml BP experiment on a GCNM Parquet root."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from types import MethodType
from pathlib import Path

import torch

from gcnm_pvi.pvi_parquet_dataset import PviParquetCompositeDataset
from gcnm_pvi.newton_coordinate_parquet_dataset import (
    PviNewtonCoordinateCombinedDataset,
)
from gcnm_pvi.newton_coordinate_hdf5_dataset import (
    PviNewtonCoordinateHdf5Dataset,
)
from gcnm_pvi.bp_artifact_gifs import generate_bp_artifact_gifs
from gcnm_pvi.determinism import configure_deterministic_training


def _coordinate_direct_process_sequence(self, sequences: dict[str, torch.Tensor]) -> torch.Tensor:
    """Assemble the selected BP tensor: literal S1, literal S2, and dS2/dt."""

    s1 = torch.nan_to_num(sequences["pviHP"], nan=self.nan_values)
    s2 = torch.nan_to_num(sequences["pviLP"], nan=self.nan_values)
    d_s2 = self._compute_diff(s2)
    return torch.cat((s1, s2, d_s2), dim=1)


def _newton_coordinate_process_sequence(
    self, sequences: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Assemble Newton's three channels followed by coordinate-direct's three."""

    hp = torch.nan_to_num(sequences["pviHP"], nan=self.nan_values)
    lp = torch.nan_to_num(sequences["pviLP"], nan=self.nan_values)
    if hp.shape[1] != 2 or lp.shape[1] != 2:
        raise ValueError("Newton+coordinate packed tensors must each have two channels")
    newton_hp, s1 = hp[:, :1], hp[:, 1:2]
    newton_lp, s2 = lp[:, :1], lp[:, 1:2]
    d_newton_lp = self._compute_diff(newton_lp)
    d2_newton_lp = self._compute_diff(d_newton_lp)
    d_s2 = self._compute_diff(s2)
    return torch.cat(
        (newton_hp, d_newton_lp, d2_newton_lp, s1, s2, d_s2), dim=1
    )


def _install_representation_preprocessor(model, manifest: dict) -> str:
    if manifest.get("schema") in {
        "pvi-newton-coordinate-paired-v1",
        "pvi-newton-coordinate-hdf5-paired-v1",
    }:
        expected = [
            "newton_hp", "d_newton_lp_dt", "d2_newton_lp_dt2",
            "s1", "s2", "d_s2_dt",
        ]
        if manifest.get("bp_channel_contract") != expected:
            raise ValueError("Newton+coordinate manifest has the wrong BP channel contract")
        model._process_sequence = MethodType(_newton_coordinate_process_sequence, model)
        return "newton_hp_dlp_ddlp_plus_s1_s2_d_s2_dt"
    if manifest.get("schema") == "pvi-gcnm-coordinate-direct-parquet-v1":
        expected = ["s1", "s2", "d_s2_dt"]
        if manifest.get("bp_channel_contract") != expected:
            raise ValueError("coordinate-direct manifest has the wrong BP channel contract")
        model._process_sequence = MethodType(_coordinate_direct_process_sequence, model)
        return "s1_s2_d_s2_dt"
    return "native_pvi_hp_dlp_ddlp"


def _mount_pvi_ml(path: Path) -> None:
    """Mount the external pvi_ml checkout under its historical ``src`` name."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        "src", path / "__init__.py", submodule_search_locations=[str(path)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pvi_ml from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["src"] = module
    spec.loader.exec_module(module)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_or_validate_contract(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(
                f"refusing to resume an experiment with a different run contract: {path}"
            )
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _migrate_legacy_auxiliary_file(legacy: Path, destination: Path) -> None:
    """Move our auxiliary JSON into a native folder and remove its old folder."""

    legacy = Path(legacy)
    destination = Path(destination)
    if not legacy.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if legacy.read_bytes() != destination.read_bytes():
            raise RuntimeError(
                f"legacy and native-layout auxiliary artifacts differ: {legacy}"
            )
        legacy.unlink()
    else:
        legacy.replace(destination)
    try:
        legacy.parent.rmdir()
    except OSError:
        pass


def _verify_native_pvi_artifacts(
    logdir: Path, subject: str, output_size: int, checkpoint: dict
) -> dict:
    """Assert compatibility with the current native ``pvi_ml`` artifact API."""

    logdir = Path(logdir)
    paths = {
        "checkpoint": logdir / "checkpoints" / f"{subject}_checkpoints.pth",
        "best_checkpoint": logdir / "checkpoints" / f"{subject}_checkpoints_best.pth",
        "config": logdir / "configs" / f"{subject}_configs.json",
        "history": logdir / "history" / f"{subject}_history.csv",
        "results": logdir / "results" / f"{subject}_results.csv",
        "statistics": logdir / "statistics" / f"{subject}_statistics.json",
    }
    # Native pvi_ml only creates the separate best file after its accuracy
    # threshold is crossed. A valid max-epoch run may therefore have the final
    # checkpoint but no best checkpoint; do not misclassify completed training.
    required_paths = {key: value for key, value in paths.items() if key != "best_checkpoint"}
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing native pvi_ml artifacts: {missing}")

    checkpoint_keys = {
        "dataset",
        "datetime",
        "epoch",
        "loss_func",
        "model",
        "name",
        "optimizer",
        "scheduler",
        "stopper",
        "tracker",
    }
    if set(checkpoint) != checkpoint_keys:
        raise RuntimeError(
            "checkpoint keys differ from native pvi_ml schema: "
            f"expected={sorted(checkpoint_keys)}, actual={sorted(checkpoint)}"
        )
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config_keys = {
        "dataset",
        "datetime",
        "environment",
        "loss_func",
        "model",
        "optimizer",
        "scheduler",
        "stopper",
        "summary",
    }
    if set(config) != config_keys:
        raise RuntimeError("config JSON differs from current native pvi_ml schema")

    with paths["history"].open(newline="", encoding="utf-8") as handle:
        history_columns = next(csv.reader(handle))
    expected_history = [
        "epoch",
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
        "lr",
    ]
    if history_columns != expected_history:
        raise RuntimeError(f"unexpected pvi_ml history columns: {history_columns}")

    with paths["results"].open(newline="", encoding="utf-8") as handle:
        result_columns = next(csv.reader(handle))
    expected_results = [
        *[f"pred_{index}" for index in range(1, output_size + 1)],
        *[f"target_{index}" for index in range(1, output_size + 1)],
    ]
    if result_columns != expected_results:
        raise RuntimeError(
            f"unexpected pvi_ml result columns for output size {output_size}"
        )

    statistics = json.loads(paths["statistics"].read_text(encoding="utf-8"))
    required_statistics = {
        "num_train",
        "num_test",
        "num_periods",
        "num_seq05",
        "dbp_mae",
        "sbp_mae",
    }
    missing_statistics = sorted(required_statistics - set(statistics))
    if missing_statistics:
        raise RuntimeError(
            f"native pvi_ml statistics are missing keys: {missing_statistics}"
        )
    return {
        "status": "pass",
        "schema": "current-pvi-ml-workflow-v3",
        "paths": {name: str(path) for name, path in paths.items()},
        "best_checkpoint_present": paths["best_checkpoint"].is_file(),
        "checkpoint_keys": sorted(checkpoint_keys),
        "history_columns": history_columns,
        "result_columns": len(result_columns),
        "statistics_keys": sorted(statistics),
    }


def _load_best_checkpoint_for_postprocess(workflow, *, device, dtype) -> tuple[dict, dict]:
    """Safely restore an intact native best checkpoint without calling initiation.

    Upstream ``initiate_training(use_checkpoint=True)`` catches every checkpoint
    load/unpack exception and then overwrites the current checkpoint with an
    epoch -1 initialization.  Postprocessing must instead fail closed and use
    the independently saved best checkpoint.
    """

    logdir = Path(workflow.path_manager.logdir)
    subject = workflow.dataset.name
    current_path = logdir / "checkpoints" / f"{subject}_checkpoints.pth"
    best_path = logdir / "checkpoints" / f"{subject}_checkpoints_best.pth"
    # A valid native run can finish without a separate ``_best`` file when
    # pvi_ml's accuracy threshold is never crossed. In that case the final
    # checkpoint is the authoritative selected model.
    source_path = best_path if best_path.is_file() else current_path
    if not source_path.is_file():
        raise FileNotFoundError(
            f"postprocess-only requires {best_path} or {current_path}"
        )
    best_hash_before = _sha256(source_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) < 0:
        raise RuntimeError(f"refusing invalid checkpoint epoch in {source_path}")
    tracker = checkpoint.get("tracker", {})
    if not tracker.get("epoch") or len(tracker.get("train_loss", [])) == 0:
        raise RuntimeError(
            f"postprocess checkpoint has no training history: {source_path}"
        )
    workflow.checkpoint.unpack(checkpoint)
    workflow.epoch = int(checkpoint["epoch"])
    workflow.status = "terminal"
    workflow.device = device
    workflow.dtype = dtype
    workflow.dataset.get_dataloaders()
    workflow.trainer = workflow.trainer.to(device=device, dtype=dtype)
    workflow.trainer.transfer_optimizer(device=device, dtype=dtype)
    if _sha256(source_path) != best_hash_before:
        raise RuntimeError("postprocess checkpoint changed while being loaded")
    return checkpoint, {
        "source": str(source_path),
        "source_sha256": best_hash_before,
        "destination": str(current_path),
        "epoch": int(checkpoint["epoch"]),
        "selection": (
            "native_best_checkpoint"
            if source_path == best_path
            else "native_final_checkpoint_no_best_threshold_crossing"
        ),
    }


def _promote_best_checkpoint(checkpoint_info: dict) -> None:
    """Make the validated best checkpoint the byte-identical native final file."""

    source = Path(checkpoint_info["source"])
    destination = Path(checkpoint_info["destination"])
    expected_hash = checkpoint_info["source_sha256"]
    if _sha256(source) != expected_hash:
        raise RuntimeError("best checkpoint changed before final promotion")
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)
    if _sha256(source) != expected_hash or _sha256(destination) != expected_hash:
        raise RuntimeError("final checkpoint promotion failed hash verification")


def _verify_inference(
    workflow,
    output_size: int,
    verification_path: Path,
    *,
    checkpoint_recovery: dict | None = None,
) -> None:
    # Upstream ``TrainingCheckpoint.load`` does not provide ``map_location``;
    # native checkpoints produced on a GPU therefore cannot be verified on a
    # CPU-only postprocessing node.  Load the same native file explicitly.
    checkpoint_path = (
        Path(workflow.path_manager.logdir)
        / "checkpoints"
        / f"{workflow.dataset.name}_checkpoints.pth"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    workflow.checkpoint.unpack(checkpoint)
    batch = next(iter(workflow.dataset.loaders["test"]))
    device = workflow.model.device
    dtype = next(workflow.model.parameters()).dtype
    batch = {
        key: value.to(device=device, dtype=dtype)
        if torch.is_tensor(value) and torch.is_floating_point(value)
        else value.to(device=device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    sequences, stats, _ = workflow.model.process_batch(batch)
    workflow.model.eval()
    with torch.inference_mode():
        prediction = workflow.model(sequences, stats)
    if tuple(prediction.shape) != (len(batch["bp"]), output_size):
        raise RuntimeError(f"unexpected inference shape {tuple(prediction.shape)}")
    if not torch.isfinite(prediction).all():
        raise RuntimeError("non-finite GCNM BP inference output")
    artifact_schema = _verify_native_pvi_artifacts(
        workflow.path_manager.logdir,
        workflow.dataset.name,
        output_size,
        checkpoint,
    )
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    verification_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "checkpoint_name": str(checkpoint["name"]),
                "test_batch_size": int(prediction.shape[0]),
                "prediction_shape": list(prediction.shape),
                "prediction_finite": True,
                "native_pvi_ml_artifacts": artifact_schema,
                "checkpoint_recovery": checkpoint_recovery,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvi-ml-root", type=Path, required=True)
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument(
        "--reference-parquet-root",
        type=Path,
        help="archived Newton image root paired with coordinate --parquet-root",
    )
    parser.add_argument(
        "--reference-hdf5-registry",
        type=Path,
        help=(
            "subject/session registry used to read archived Newton images directly "
            "from source HDF5 for six-channel training"
        ),
    )
    parser.add_argument(
        "--gif-reference-parquet-root",
        type=Path,
        help=(
            "reference-image Parquet used only for post-training PVI/Newton GIF panels; "
            "defaults to --reference-parquet-root for 6-channel runs"
        ),
    )
    parser.add_argument(
        "--gif-reference-hdf5-registry",
        type=Path,
        help="source-HDF5 registry used by prediction-aligned GIF rendering",
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--family",
        choices=["coordinate", "global_voltage_slots", "newton_coordinate"],
        required=True,
    )
    parser.add_argument("--architecture", choices=["crt", "crs"], required=True)
    parser.add_argument("--output-mode", choices=["waveform", "fiducials"], required=True)
    parser.add_argument("--channel-mode", choices=["3ch", "6ch"], default="3ch")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use deterministic CUDA and a seeded DataLoader permutation",
    )
    parser.add_argument(
        "--defer-artifact-gifs",
        action="store_true",
        help="leave GIF rendering to a dependent CPU postprocessing array",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="load an existing checkpoint and only export/verify final artifacts",
    )
    args = parser.parse_args()

    _mount_pvi_ml(args.pvi_ml_root)
    from src.models.attn_models import PviCNNTransformer
    from src.models.early_stopper import EarlyStopper
    from src.models.loss_functions import MorphologyLoss
    from src.models.s4_models import PviSamba
    from src.models.workflow_v3 import TrainingWorkflow
    from src.pipeline.data_discovery import ProjectPathManager
    from src.utils.primitives import DEFAULT_TRAIN_DEVICE, DEFAULT_TRAIN_DTYPE

    if (
        args.reference_parquet_root is not None
        and args.reference_hdf5_registry is not None
    ):
        raise ValueError(
            "use either --reference-parquet-root or --reference-hdf5-registry"
        )
    if args.reference_hdf5_registry is not None:
        if args.family != "newton_coordinate" or args.channel_mode != "6ch":
            raise ValueError(
                "HDF5 Newton+coordinate input requires family=newton_coordinate and 6ch"
            )
        dataset = PviNewtonCoordinateHdf5Dataset(
            args.reference_hdf5_registry,
            args.parquet_root,
            args.subject,
            args.output_mode,
            args.split_manifest,
            name=args.subject,
        ).build()
    elif args.reference_parquet_root is not None:
        if args.family != "newton_coordinate" or args.channel_mode != "6ch":
            raise ValueError("paired Newton+coordinate input requires family=newton_coordinate and 6ch")
        dataset = PviNewtonCoordinateCombinedDataset(
            args.reference_parquet_root,
            args.parquet_root,
            args.subject,
            args.output_mode,
            args.split_manifest,
            name=args.subject,
        ).build()
    else:
        if args.family == "newton_coordinate":
            raise ValueError(
                "newton_coordinate requires a Parquet or HDF5 Newton reference"
            )
        dataset = PviParquetCompositeDataset(
            args.parquet_root,
            subjects=[args.subject],
            input_mode="image",
            output_mode=args.output_mode,
            mask_key="mask05",
            channel_mode=args.channel_mode,
            split_manifest=args.split_manifest,
            name=args.subject,
        ).build()
    dataset.set_partition(test_size=0.1, shuffle=True, random_state=42)
    dataset.get_partition()
    desired_train_ids = dataset.state_dict()["train_sample_ids"]
    desired_test_ids = dataset.state_dict()["test_sample_ids"]
    loader_generator = (
        configure_deterministic_training(args.seed)
        if args.deterministic
        else torch.Generator().manual_seed(args.seed)
    )
    loader_options = {
        "batch_size": 32,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": loader_generator,
    }
    if args.num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    dataset.set_dataloaders(
        **loader_options,
    )
    model_class = PviCNNTransformer if args.architecture == "crt" else PviSamba
    model = model_class(dataset.shapes)
    preprocessing_contract = _install_representation_preprocessor(model, dataset.manifest)
    loss = MorphologyLoss(base_loss=torch.nn.MSELoss(), base_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, mode="min", factor=0.9
    )
    stopper = EarlyStopper(
        patience=200, delta=1e-4, mode="max", threshold=0.5, verbose=True
    )
    target = (
        f"gcnm-{args.family}-{args.channel_mode}-{args.architecture}"
        f"-image-to-{args.output_mode}"
    )
    manager = ProjectPathManager(branch="main", target=target, export_root=args.artifact_root)
    gif_reference_root = (
        args.gif_reference_parquet_root or args.reference_parquet_root
    )
    gif_reference_registry = (
        args.gif_reference_hdf5_registry or args.reference_hdf5_registry
    )
    representation_manifest = args.parquet_root / "manifest.json"
    representation_schema = json.loads(
        representation_manifest.read_text(encoding="utf-8")
    ).get("schema")
    make_coordinate_gifs = representation_schema == "pvi-gcnm-coordinate-direct-parquet-v1"
    if (
        make_coordinate_gifs
        and gif_reference_root is None
        and gif_reference_registry is None
    ):
        raise ValueError(
            "coordinate BP artifacts require a Parquet or HDF5 Newton GIF reference"
        )
    contract = {
        "schema": "pvi-bp-run-contract-v1",
        "representation": args.family,
        "channel_mode": args.channel_mode,
        "subject": args.subject,
        "architecture": args.architecture,
        "output_mode": args.output_mode,
        "mask_key": "mask05",
        "preprocessing_contract": preprocessing_contract,
        "parquet_root": str(args.parquet_root.resolve()),
        "representation_manifest_sha256": _sha256(representation_manifest),
        "reference_parquet_root": (
            None
            if args.reference_parquet_root is None
            else str(args.reference_parquet_root.resolve())
        ),
        "reference_manifest_sha256": (
            None
            if args.reference_parquet_root is None
            else _sha256(args.reference_parquet_root / "manifest.json")
        ),
        "reference_hdf5_registry": (
            None
            if args.reference_hdf5_registry is None
            else str(args.reference_hdf5_registry.resolve())
        ),
        "reference_hdf5_registry_sha256": (
            None
            if args.reference_hdf5_registry is None
            else _sha256(args.reference_hdf5_registry)
        ),
        "gif_reference_hdf5_registry": (
            None
            if gif_reference_registry is None
            else str(gif_reference_registry.resolve())
        ),
        "gif_reference_hdf5_registry_sha256": (
            None
            if gif_reference_registry is None
            else _sha256(gif_reference_registry)
        ),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": _sha256(args.split_manifest),
        "split_counts": {
            "train": len(desired_train_ids),
            "test": len(desired_test_ids),
            "excluded": len(dataset) - len(desired_train_ids) - len(desired_test_ids),
        },
        "protocol": {
            "batch_size": 32,
            "morphology_mse_weight": 0.2,
            "optimizer": "AdamW",
            "learning_rate": 5e-4,
            "weight_decay": 1e-2,
            "scheduler_patience": 100,
            "scheduler_factor": 0.9,
            "early_stopping_patience": 200,
            "early_stopping_delta": 1e-4,
            "early_stopping_threshold": 0.5,
            "checkpoint_interval_epochs": 100,
            "maximum_epochs": args.max_epochs,
            "seed": args.seed,
            "deterministic_algorithms": args.deterministic,
            "dataloader_workers": args.num_workers,
        },
    }
    native_contract = manager.logdir / "configs" / f"{args.subject}_run_contract.json"
    _migrate_legacy_auxiliary_file(
        manager.logdir / "run_contracts" / f"{args.subject}.json",
        native_contract,
    )
    _write_or_validate_contract(
        native_contract, contract
    )
    workflow = TrainingWorkflow(
        path_manager=manager,
        dataset=dataset,
        model=model,
        loss_func=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        stopper=stopper,
    )
    workflow.set_checkpoint_interval(epochs=100)
    checkpoint_recovery = None
    if args.postprocess_only:
        _, checkpoint_recovery = _load_best_checkpoint_for_postprocess(
            workflow, device="cpu", dtype=DEFAULT_TRAIN_DTYPE
        )
    else:
        workflow.initiate_training(
            use_checkpoint=True, device=DEFAULT_TRAIN_DEVICE, dtype=DEFAULT_TRAIN_DTYPE
        )
    resumed = dataset.state_dict()
    if (
        resumed["train_sample_ids"] != desired_train_ids
        or resumed["test_sample_ids"] != desired_test_ids
    ):
        raise RuntimeError("resumed checkpoint does not use the frozen pilot split")
    if not args.postprocess_only:
        workflow.run(min_epochs=1, max_epochs=args.max_epochs)
        workflow.checkpoint.save()
    else:
        _promote_best_checkpoint(checkpoint_recovery)
    # pvi_ml's eager HDF5 dataset is moved to the model device, but a lazy
    # Parquet dataset must keep returning CPU tensors (especially from worker
    # processes).  ArtifactEvaluator does not transfer its batches, so perform
    # this final, small evaluation on CPU.  GPU training/checkpoint selection is
    # already complete and the saved weights are unchanged.
    # Move the handler, not only its model. BaseModelHandler.evaluate_epoch()
    # transfers every batch to ``trainer.device``; leaving that field as CUDA
    # while moving only the weights to CPU causes a post-training device
    # mismatch during results/statistics export.
    workflow.trainer = workflow.trainer.to(
        device="cpu", dtype=DEFAULT_TRAIN_DTYPE
    )
    workflow.model = workflow.trainer.model
    workflow.dataset = workflow.trainer.dataset
    workflow.device = "cpu"
    workflow.dtype = DEFAULT_TRAIN_DTYPE
    # ``terminate_training`` may restore an earlier best checkpoint while
    # leaving predictions from the final pre-restore epoch in this cache.
    # Force result CSV/statistics/GIFs to evaluate the selected checkpoint.
    workflow.trainer.eval_results.clear()
    # Explicit terminal status is required by checkpoint-only postprocessing;
    # otherwise upstream exports config/history/results but silently omits the
    # statistics artifact.
    workflow.export_artifacts(status="terminal")
    verification_path = manager.logdir / "gifs" / f"{args.subject}_verification.json"
    _migrate_legacy_auxiliary_file(
        manager.logdir / "verification" / f"{args.subject}.json",
        verification_path,
    )
    _verify_inference(
        workflow,
        50 if args.output_mode == "waveform" else 2,
        verification_path,
        checkpoint_recovery=checkpoint_recovery,
    )
    if make_coordinate_gifs and not args.defer_artifact_gifs:
        generate_bp_artifact_gifs(
            artifact_main=manager.logdir,
            subject=args.subject,
            output_mode=args.output_mode,
            coordinate_root=args.parquet_root,
            reference_root=gif_reference_root,
            reference_registry=gif_reference_registry,
            split_manifest=args.split_manifest,
        )


if __name__ == "__main__":
    main()
