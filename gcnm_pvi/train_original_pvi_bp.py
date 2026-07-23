"""Train one split-matched Newton-image BP baseline with unmodified pvi_ml."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

from gcnm_pvi.pvi_splits import stable_sample_id, validate_split
from gcnm_pvi.determinism import configure_deterministic_training


PILOT_SESSIONS = ("baseline", "valsalva", "pressor")


def _mount_pvi_ml(path: Path) -> None:
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _subject_records(registry_path: Path, subject: str) -> list[dict]:
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    records = [
        record
        for record in registry["records"]
        if str(record["subject"]).lower() == subject.lower()
        and not record.get("exclusion_reason")
    ]
    records.sort(key=lambda record: int(record["source_order"]))
    sessions = tuple(str(record["session"]) for record in records)
    if sessions != PILOT_SESSIONS:
        raise ValueError(
            f"{subject} must contain exactly {PILOT_SESSIONS}; found {sessions}"
        )
    for record in records:
        path = Path(record["source_hdf5"])
        if not path.is_file():
            raise FileNotFoundError(path)
    return records


def _apply_frozen_split(dataset, split_path: Path) -> dict:
    manifest = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if manifest.get("mask_key") != "mask05":
        raise ValueError("frozen split manifest must use mask05")
    assignments = manifest.get("assignments", {})
    rows: list[dict] = []
    train_mask: list[tuple[int, int]] = []
    test_mask: list[tuple[int, int]] = []
    offset = 0
    for raw in dataset.raws:
        for local_start, local_stop in raw.masks["mask05"]:
            sample_id = stable_sample_id(raw.name, "mask05", local_start, local_stop)
            if sample_id not in assignments:
                raise ValueError(
                    f"split manifest has no assignment for {raw.name} "
                    f"mask05 [{local_start}, {local_stop})"
                )
            label = assignments[sample_id]
            global_mask = (int(local_start + offset), int(local_stop + offset))
            if label == "train":
                train_mask.append(global_mask)
            elif label == "test":
                test_mask.append(global_mask)
            elif label != "excluded":
                raise ValueError(f"invalid split label {label!r} for {sample_id}")
            rows.append(
                {
                    "sample_id": sample_id,
                    "subject": raw.subject,
                    "session": raw.session,
                    "source_name": raw.name,
                    "mask_start": int(local_start),
                    "mask_stop": int(local_stop),
                }
            )
        offset += int(raw.num_periods)

    validate_split(rows, assignments)
    if not train_mask or not test_mask:
        raise ValueError("frozen split must have non-empty train and test partitions")
    expected_active = set(train_mask) | set(test_mask)
    if not expected_active <= set(dataset.active_mask):
        raise ValueError("frozen split masks are not present in the Newton dataset")
    dataset.train_mask = sorted(train_mask)
    dataset.test_mask = sorted(test_mask)
    dataset.subsets = dataset._get_subsets_from_split()
    return {
        "train": dataset.train_mask.copy(),
        "test": dataset.test_mask.copy(),
        "excluded": len(rows) - len(train_mask) - len(test_mask),
        "sample_count": len(rows),
    }


def _write_or_validate_contract(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"refusing to resume an experiment with a different run contract: {path}"
            )
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _verify_inference(workflow, output_size: int, verification_path: Path) -> None:
    checkpoint = workflow.checkpoint.load()
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
        raise RuntimeError("non-finite Newton baseline inference output")
    report = {
        "status": "pass",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_name": str(checkpoint["name"]),
        "test_batch_size": int(prediction.shape[0]),
        "prediction_shape": list(prediction.shape),
        "prediction_finite": True,
    }
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    verification_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvi-ml-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--subject", choices=["subject006", "subject010"], required=True)
    parser.add_argument("--input-mode", choices=["img", "bioz"], default="img")
    parser.add_argument("--architecture", choices=["crt", "crs"], required=True)
    parser.add_argument("--output-mode", choices=["waveform", "fiducials"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="enforce deterministic CUDA and use a seeded DataLoader permutation",
    )
    parser.add_argument(
        "--target",
        help="native pvi_ml model-folder name; defaults to a mapper-compatible name",
    )
    parser.add_argument(
        "--native-artifacts-only",
        action="store_true",
        help="write only the five directories created by pvi_ml ProjectPathManager",
    )
    args = parser.parse_args()
    if args.num_workers != 0:
        raise ValueError(
            "native eager pvi_ml tensors are transferred to CUDA before iteration; "
            "DataLoader multiprocessing must remain disabled"
        )

    records = _subject_records(args.registry, args.subject)
    split_hash = _sha256(args.split_manifest)
    source_files = {}
    for record in records:
        source_path = Path(record["source_hdf5"])
        source_stat = source_path.stat()
        source_files[record["source_name"]] = {
            "path": str(source_path.resolve()),
            "size_bytes": int(source_stat.st_size),
            "mtime_ns": int(source_stat.st_mtime_ns),
        }

    _mount_pvi_ml(args.pvi_ml_root)
    from src.models.attn_models import PviCNNTransformer
    from src.models.early_stopper import EarlyStopper
    from src.models.loss_functions import MorphologyLoss
    from src.models.s4_models import PviSamba
    from src.models.workflow_v3 import TrainingWorkflow
    from src.pipeline.data_discovery import ProjectPathManager
    from src.pipeline.data_extraction import PviRawDataset
    from src.pipeline.data_preparation_eager import PviCompositeDataset
    from src.utils.primitives import (
        DEFAULT_TRAIN_DEVICE,
        DEFAULT_TRAIN_DTYPE,
        PviDataFile,
    )

    data_files = [
        PviDataFile(
            name=str(record["source_name"]),
            subject=str(record["subject"]),
            session=str(record["session"]),
            path=Path(record["source_hdf5"]),
        )
        for record in records
    ]
    raws = [PviRawDataset(ds_file=data_file).load() for data_file in data_files]
    dataset = PviCompositeDataset(
        ds_raws=raws,
        input_mode=args.input_mode,
        output_mode=args.output_mode,
        mask_key="mask05",
        name=args.subject,
    ).build(cleanup=True)
    desired_split = _apply_frozen_split(dataset, args.split_manifest)
    if args.deterministic and args.seed is None:
        raise ValueError("--deterministic requires --seed")
    loader_generator = (
        configure_deterministic_training(args.seed) if args.deterministic else None
    )
    loader_options = {
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 0,
    }
    if loader_generator is not None:
        loader_options["generator"] = loader_generator
    dataset.set_dataloaders(**loader_options)

    model_class = PviCNNTransformer if args.architecture == "crt" else PviSamba
    model = model_class(dataset.shapes)
    loss = MorphologyLoss(base_loss=torch.nn.MSELoss(), base_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, mode="min", factor=0.9
    )
    stopper = EarlyStopper(
        patience=200, delta=1e-4, mode="max", threshold=0.5, verbose=True
    )
    target = args.target or (
        f"{args.subject}-{args.architecture}-{args.input_mode}-to-{args.output_mode}"
    )
    expected_target = (
        f"{args.subject}-{args.architecture}-{args.input_mode}-to-{args.output_mode}"
    )
    if args.target is not None and args.target != expected_target:
        raise ValueError(
            "target must retain the native pvi_ml mapper contract; expected "
            f"{expected_target!r}"
        )
    target_root = args.artifact_root / target
    if target_root.exists():
        raise FileExistsError(f"refusing to resume/merge strict native baseline: {target_root}")
    manager = ProjectPathManager(
        branch="main", target=target, export_root=args.artifact_root
    )
    contract = {
        "schema": "pvi-bp-run-contract-v1",
        "representation": "archived-newton-pvi",
        "subject": args.subject,
        "sessions": list(PILOT_SESSIONS),
        "architecture": args.architecture,
        "output_mode": args.output_mode,
        "mask_key": "mask05",
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": split_hash,
        "registry": str(args.registry.resolve()),
        "registry_sha256": _sha256(args.registry),
        # The immutable Parquet manifests retain full HDF5 hashes. Avoid
        # re-reading multi-GB source files independently in every BP task.
        "source_hdf5": source_files,
        "split_counts": {
            "train": len(desired_split["train"]),
            "test": len(desired_split["test"]),
            "excluded": desired_split["excluded"],
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
        },
    }
    if args.seed is not None or args.deterministic:
        contract["protocol"].update(
            seed=args.seed,
            deterministic_algorithms=args.deterministic,
            deterministic_unsupported_cuda="warn_only",
            dataloader_workers=0,
        )
    if not args.native_artifacts_only:
        _write_or_validate_contract(
            manager.logdir / "run_contracts" / f"{args.subject}.json", contract
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
    workflow.initiate_training(
        use_checkpoint=True, device=DEFAULT_TRAIN_DEVICE, dtype=DEFAULT_TRAIN_DTYPE
    )
    if dataset.train_mask != desired_split["train"] or dataset.test_mask != desired_split["test"]:
        raise RuntimeError("resumed checkpoint does not use the frozen pilot split")
    workflow.run(min_epochs=1, max_epochs=args.max_epochs)
    workflow.checkpoint.save()
    # The workflow can restore an earlier best checkpoint at termination while
    # retaining final-epoch predictions. Re-evaluate the selected checkpoint.
    workflow.trainer.eval_results.clear()
    workflow.export_artifacts()
    if not args.native_artifacts_only:
        output_size = 50 if args.output_mode == "waveform" else 2
        verification = manager.logdir / "verification" / f"{args.subject}.json"
        verification.parent.mkdir(parents=True, exist_ok=True)
        _verify_inference(workflow, output_size, verification)


if __name__ == "__main__":
    main()
