"""Train one original-protocol CRT from archived reference Parquet inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gcnm_pvi.reference_parquet_dataset import PviReferenceParquetDataset
from gcnm_pvi.determinism import configure_deterministic_training
from gcnm_pvi.train_pvi_bp import (
    _load_best_checkpoint_for_postprocess,
    _mount_pvi_ml,
    _promote_best_checkpoint,
    _sha256,
    _verify_inference,
    _write_or_validate_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvi-ml-root", type=Path, required=True)
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--subject", choices=["subject006", "subject010"], required=True)
    parser.add_argument("--input-mode", choices=["img", "bioz"], required=True)
    parser.add_argument("--output-mode", choices=["waveform", "fiducials"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="enforce deterministic CUDA and use a seeded DataLoader permutation",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="load a finalized checkpoint and only export/verify its artifacts",
    )
    args = parser.parse_args()

    _mount_pvi_ml(args.pvi_ml_root)
    from src.models.attn_models import PviCNNTransformer
    from src.models.early_stopper import EarlyStopper
    from src.models.loss_functions import MorphologyLoss
    from src.models.workflow_v3 import TrainingWorkflow
    from src.pipeline.data_discovery import ProjectPathManager
    from src.utils.primitives import DEFAULT_TRAIN_DEVICE, DEFAULT_TRAIN_DTYPE

    dataset = PviReferenceParquetDataset(
        args.parquet_root,
        subject=args.subject,
        output_mode=args.output_mode,
        split_manifest=args.split_manifest,
    ).build()
    manifest_mode = dataset.manifest["input_mode"]
    if manifest_mode != args.input_mode:
        raise ValueError(f"Parquet input_mode {manifest_mode!r} != requested {args.input_mode!r}")
    dataset.set_partition(test_size=0.1, shuffle=True, random_state=42)
    dataset.get_partition()
    desired_train_ids = dataset.state_dict()["train_sample_ids"]
    desired_test_ids = dataset.state_dict()["test_sample_ids"]
    if args.deterministic and args.seed is None:
        raise ValueError("--deterministic requires --seed")
    loader_generator = (
        configure_deterministic_training(args.seed) if args.deterministic else None
    )
    loader_options = {
        "batch_size": 32,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    if loader_generator is not None:
        loader_options["generator"] = loader_generator
    dataset.set_dataloaders(**loader_options)

    model = PviCNNTransformer(dataset.shapes)
    loss = MorphologyLoss(base_loss=torch.nn.MSELoss(), base_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, mode="min", factor=0.9
    )
    stopper = EarlyStopper(
        patience=200, delta=1e-4, mode="max", threshold=0.5, verbose=True
    )
    target = f"reference-parquet-crt-{args.input_mode}-to-{args.output_mode}"
    manager = ProjectPathManager(branch="main", target=target, export_root=args.artifact_root)
    contract = {
        "schema": "pvi-bp-run-contract-v1",
        "representation": dataset.manifest["representation"],
        "storage": "parquet",
        "subject": args.subject,
        "architecture": "crt",
        "input_mode": args.input_mode,
        "output_mode": args.output_mode,
        "mask_key": "mask05",
        "parquet_root": str(args.parquet_root.resolve()),
        "representation_manifest_sha256": _sha256(args.parquet_root / "manifest.json"),
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
        },
    }
    if args.seed is not None or args.deterministic:
        contract["protocol"].update(
            seed=args.seed,
            deterministic_algorithms=args.deterministic,
            deterministic_unsupported_cuda="warn_only",
            dataloader_workers=args.num_workers,
        )
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
    checkpoint_recovery = None
    if args.postprocess_only:
        _, checkpoint_recovery = _load_best_checkpoint_for_postprocess(
            workflow, device="cpu", dtype=DEFAULT_TRAIN_DTYPE
        )
    else:
        workflow.initiate_training(
            use_checkpoint=True,
            device=DEFAULT_TRAIN_DEVICE,
            dtype=DEFAULT_TRAIN_DTYPE,
        )
    resumed = dataset.state_dict()
    if resumed["train_sample_ids"] != desired_train_ids or resumed["test_sample_ids"] != desired_test_ids:
        raise RuntimeError("resumed checkpoint does not preserve frozen sample-ID split")
    if not args.postprocess_only:
        workflow.run(min_epochs=1, max_epochs=args.max_epochs)
        workflow.checkpoint.save()
    else:
        _promote_best_checkpoint(checkpoint_recovery)
    # The upstream artifact evaluator assumes eager datasets were moved to the
    # model device.  Lazy Parquet workers return CPU tensors, so evaluate the
    # finalized checkpoint on CPU after GPU training has finished.
    workflow.model.to(device="cpu")
    # Recompute artifacts from the selected checkpoint, not cached predictions
    # from the last epoch before best-checkpoint restoration.
    workflow.trainer.eval_results.clear()
    # A checkpoint-only workflow has not traversed ``run()``, so upstream keeps
    # a non-terminal in-memory status and silently skips statistics unless the
    # finalized status is explicit.  This does not modify model weights.
    workflow.export_artifacts(status="terminal")
    _verify_inference(
        workflow,
        50 if args.output_mode == "waveform" else 2,
        manager.logdir / "verification" / f"{args.subject}.json",
        checkpoint_recovery=checkpoint_recovery,
    )


if __name__ == "__main__":
    main()
