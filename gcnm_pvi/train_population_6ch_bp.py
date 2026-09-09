"""Train PVI image-to-waveform CRT or Samba on six GCNM image channels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from types import MethodType

import torch

from gcnm_pvi.determinism import configure_deterministic_training
from gcnm_pvi.population_6ch_parquet import PopulationSixChannelParquetDataset


EXPERIMENTS = {
    "crt": {"architecture": "crt", "disjoint_id": "pd13"},
    "samba": {"architecture": "samba", "disjoint_id": "pd17"},
}


def experiment_target(experiment: str, split_mode: str) -> str:
    if split_mode == "within":
        return f"{experiment}-gcnm6ch-img-to-waveform-exact-pvi"
    if split_mode == "disjoint":
        identifier = EXPERIMENTS[experiment]["disjoint_id"]
        return f"{identifier}-{experiment}-gcnm6ch-img-to-waveform-exact-pvi"
    raise ValueError(f"unsupported population split mode: {split_mode}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def promote_checkpoint_file(
    source: Path,
    destination: Path,
    *,
    expected_epoch: int | None = None,
) -> dict:
    """Atomically promote a selected PVI checkpoint to its standard path."""

    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    epoch = int(payload["epoch"])
    if expected_epoch is not None and epoch != int(expected_epoch):
        raise RuntimeError(
            f"selected checkpoint epoch differs: {epoch} != {int(expected_epoch)}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if _sha256(destination) != _sha256(source):
        raise RuntimeError("promoted checkpoint checksum differs from selected checkpoint")
    promoted = torch.load(destination, map_location="cpu", weights_only=True)
    if int(promoted["epoch"]) != epoch:
        raise RuntimeError("promoted checkpoint epoch differs after atomic replacement")
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "epoch": epoch,
        "sha256": _sha256(destination),
    }


def finalize_selected_checkpoint(workflow) -> dict:
    """Persist the checkpoint that PVI's termination logic selected in memory.

    Upstream ``TrainingWorkflow.terminate_training`` restores the best
    checkpoint into the workflow, but it does not save that restored state to
    the standard checkpoint path used by ``inference_population.py``.  Keep the
    upstream training behavior unchanged and repair only that artifact handoff.
    """

    manager = workflow.path_manager
    dataset_name = workflow.dataset.name
    current_path = manager.generate_artifact_path(
        core_name=dataset_name,
        artifact_name="checkpoints",
        extension="pth",
    )
    best_path = manager.generate_artifact_path(
        core_name=dataset_name,
        artifact_name="checkpoints",
        suffix="best",
        extension="pth",
    )
    selected_epoch = int(workflow.epoch)
    if best_path.is_file():
        best = torch.load(best_path, map_location="cpu", weights_only=True)
    else:
        best = None

    if best is not None and int(best["epoch"]) == selected_epoch:
        report = promote_checkpoint_file(
            best_path,
            current_path,
            expected_epoch=selected_epoch,
        )
        report["selection"] = "best"
    else:
        # The terminal in-memory state beat the previous best (or no best was
        # available). Persist it to both standard checkpoint names.
        selected = workflow.checkpoint.create(name="terminal")
        if int(selected["epoch"]) != selected_epoch:
            raise RuntimeError(
                "terminal checkpoint epoch differs from workflow-selected epoch"
            )
        temporary = current_path.with_name(f".{current_path.name}.{os.getpid()}.tmp")
        try:
            torch.save(selected, temporary)
            os.replace(temporary, current_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        report = promote_checkpoint_file(
            current_path,
            best_path,
            expected_epoch=selected_epoch,
        )
        report["source"] = str(current_path.resolve())
        report["destination"] = str(current_path.resolve())
        report["selection"] = "terminal"

    print(
        "Final checkpoint promotion: "
        f"selection={report['selection']}, epoch={report['epoch']}, "
        f"path={current_path}"
    )
    # Persist lightweight metadata from the same selected in-memory state.
    # Do not call TrainingWorkflow.export_artifacts(status="terminal") here:
    # its population statistics path attempts the quadratic Wasserstein
    # calculation. Dedicated inference writes results/statistics afterward.
    config_path = manager.generate_artifact_path(
        core_name=dataset_name,
        artifact_name="configs",
        extension="json",
    )
    history_path = manager.generate_artifact_path(
        core_name=dataset_name,
        artifact_name="history",
        extension="csv",
    )
    workflow.status = "terminal"
    workflow.logger.update(status="terminal")
    workflow.logger.export(config_path)
    workflow.tracker.export(history_path)
    report["config"] = str(config_path.resolve())
    report["history"] = str(history_path.resolve())
    return report


def mount_pvi_ml(path: Path) -> None:
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        "src", path / "__init__.py", submodule_search_locations=[str(path)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pvi_ml from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["src"] = module
    spec.loader.exec_module(module)


def six_channel_process_sequence(self, sequences: dict[str, torch.Tensor]) -> torch.Tensor:
    """Create the exact six image channels used by the subject-specific runs."""

    hp = torch.nan_to_num(sequences["pviHP"], nan=self.nan_values)
    lp = torch.nan_to_num(sequences["pviLP"], nan=self.nan_values)
    if hp.shape[1] != 2 or lp.shape[1] != 2:
        raise ValueError(f"expected two HP and two LP base channels, got {hp.shape} and {lp.shape}")
    newton_hp, s1 = hp[:, :1], hp[:, 1:2]
    newton_lp, s2 = lp[:, :1], lp[:, 1:2]
    d_newton_lp = self._compute_diff(newton_lp)
    d2_newton_lp = self._compute_diff(d_newton_lp)
    d_s2 = self._compute_diff(s2)
    return torch.cat((newton_hp, d_newton_lp, d2_newton_lp, s1, s2, d_s2), dim=1)


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=500)
    args = parser.parse_args()

    config = EXPERIMENTS[args.experiment]
    mount_pvi_ml(args.pvi_ml_root)
    from src.models.attn_models import PviCNNTransformer
    from src.models.early_stopper import EarlyStopper
    from src.models.loss_functions import MorphologyLoss
    from src.models.s4_models import PviSamba
    from src.models.workflow_v3 import TrainingWorkflow
    from src.pipeline.data_discovery import ProjectPathManager
    from src.utils.primitives import DEFAULT_TRAIN_DEVICE, DEFAULT_TRAIN_DTYPE

    configure_deterministic_training(args.seed)
    dataset = PopulationSixChannelParquetDataset(
        args.cache_root,
        seed=args.seed,
        require_exact_pvi_schedules=True,
    ).build()
    if dataset.split_mode != args.split_mode:
        raise ValueError(
            f"cache split mode differs: {dataset.split_mode} != {args.split_mode}"
        )
    schedule_epochs = int(dataset.manifest["batch_layout"]["epochs"])
    required_schedule_epochs = int(args.max_epochs) + 1
    if schedule_epochs < required_schedule_epochs:
        raise ValueError(
            "exact PVI cache does not reserve a terminal evaluation schedule: "
            f"available={schedule_epochs}, required={required_schedule_epochs}"
        )
    dataset.set_partition(
        test_size=0.1,
        shuffle=True,
        split_mode=args.split_mode,
        random_state=args.seed,
    )
    dataset.get_partition()
    dataset.set_dataloaders(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2,
    )

    model_class = PviCNNTransformer if config["architecture"] == "crt" else PviSamba
    model = model_class(dataset.shapes)
    model._process_sequence = MethodType(six_channel_process_sequence, model)
    loss = MorphologyLoss(base_loss=torch.nn.MSELoss(), base_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, mode="min", factor=0.8
    )
    stopper = EarlyStopper(
        patience=50, delta=1e-4, mode="max", threshold=0.5, verbose=True
    )

    target = experiment_target(args.experiment, args.split_mode)
    manager = ProjectPathManager(branch="main", target=target, export_root=args.artifact_root)
    # Gifs are produced by the normal post-training artifact pass.  Keep the
    # directory present from the beginning so the tree matches PVI exports.
    (manager.logdir / "gifs").mkdir(parents=True, exist_ok=True)

    workflow = TrainingWorkflow(
        path_manager=manager,
        dataset=dataset,
        model=model,
        loss_func=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        stopper=stopper,
    )
    workflow.set_checkpoint_interval(minutes=120, epochs=10)
    workflow.initiate_training(
        use_checkpoint=True,
        device=DEFAULT_TRAIN_DEVICE,
        dtype=DEFAULT_TRAIN_DTYPE,
    )
    workflow.run(min_epochs=1, max_epochs=args.max_epochs)
    finalize_selected_checkpoint(workflow)


if __name__ == "__main__":
    main()
