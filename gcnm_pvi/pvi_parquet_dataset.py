"""GCNM Parquet loader for HP/LP and coordinate-direct representations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from gcnm_pvi.pvi_splits import deterministic_subject_split, validate_split


def _enum_value(value) -> str:
    return str(getattr(value, "value", value)).lower()


class _ModeValue(str):
    """String-compatible mode with the ``.value`` API expected by pvi_ml."""

    @property
    def value(self) -> str:
        return str(self)


class _ParquetSplit(Dataset):
    def __init__(self, owner: "PviParquetCompositeDataset", indices: list[int]) -> None:
        self.owner = owner
        self.indices = indices
        self.input_mode = owner.input_mode
        self.output_mode = owner.output_mode
        self.mask_key = owner.mask_key

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.owner._sample(self.indices[index])


class PviParquetCompositeDataset(Dataset):
    """Subject-filterable replacement for ``PviCompositeDataset``.

    The stored BP column is always the original 50-point waveform. Fiducials
    are derived at read time as ``[min, max]``, exactly matching pvi_ml.
    """

    def __init__(
        self,
        root: str | Path,
        subjects: str | Iterable[str] | None,
        input_mode="image",
        output_mode="waveform",
        mask_key="mask05",
        channel_mode="3ch",
        split_manifest: str | Path | dict | None = None,
        name: str = "gcnm_parquet",
        verbose: bool = True,
    ) -> None:
        image_mode = _enum_value(input_mode)
        if image_mode not in {"image", "img", "3d"}:
            raise ValueError("GCNM Parquet representations support input_mode='image' only")
        output = _enum_value(output_mode)
        aliases = {"minmax": "fiducials", "bp": "waveform"}
        output = aliases.get(output, output)
        if output not in {"waveform", "fiducials"}:
            raise ValueError("output_mode must be waveform or fiducials")
        if _enum_value(mask_key).replace("_", "") not in {"mask05", "seq05"}:
            raise ValueError("this representation was exported for mask05")
        channels = _enum_value(channel_mode).replace("-", "").replace("_", "")
        aliases_channels = {"3": "3ch", "three": "3ch", "6": "6ch", "six": "6ch"}
        channels = aliases_channels.get(channels, channels)
        if channels not in {"3ch", "6ch"}:
            raise ValueError("channel_mode must be '3ch' or '6ch'")

        self.root = Path(root)
        self.name = name
        self.path = str(self.root)
        self.input_mode = _ModeValue("img")
        self.output_mode = _ModeValue(output)
        self.mask_key = _ModeValue("mask05")
        self._output_value = output
        self.channel_mode = channels
        self._verbose = verbose
        self.subjects = (
            None
            if subjects is None
            else {subjects.lower()} if isinstance(subjects, str)
            else {str(subject).lower() for subject in subjects}
        )
        self._split_manifest_input = split_manifest
        self.manifest: dict = {}
        self.table = None
        self.metadata_rows: list[dict] = []
        self.subsets: dict[str, Dataset] = {}
        self.loaders: dict[str, DataLoader] = {}
        self.active_mask: list[tuple[int, int]] = []
        self.train_mask: list[tuple[int, int]] = []
        self.test_mask: list[tuple[int, int]] = []
        self._loader_params: dict = {}
        self._split_params: dict = {}
        self.period_length = 50
        self.num_periods = 0
        self.raws = []

    def build(self, cleanup: bool = True) -> "PviParquetCompositeDataset":
        import pyarrow.dataset as pads

        manifest_path = self.root / "manifest.json"
        if (self.root / "_INCOMPLETE").exists():
            raise RuntimeError(f"representation export is incomplete: {self.root}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"representation manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        supported_schemas = {
            "pvi-gcnm-hp-lp-parquet-v3",
            "pvi-gcnm-coordinate-direct-parquet-v1",
        }
        if self.manifest.get("schema") not in supported_schemas:
            raise ValueError(
                "representation must use a supported GCNM Parquet schema"
            )
        if self.manifest.get("mask_key") != "mask05":
            raise ValueError("representation manifest was not built from mask05")
        dataset = pads.dataset(str(self.root / "shards"), format="parquet")
        predicate = None
        if self.subjects:
            predicate = pads.field("subject").isin(sorted(self.subjects))
        self.table = dataset.to_table(filter=predicate).sort_by(
            [("source_order", "ascending"), ("mask_start", "ascending"), ("mask_stop", "ascending")]
        )
        metadata = self.table.select(
            ["sample_id", "subject", "session", "source_name", "mask_start", "mask_stop", "num_periods"]
        )
        self.metadata_rows = metadata.to_pylist()
        if not self.metadata_rows:
            raise ValueError("subject filter selected zero representation rows")
        self.active_mask = [(index, index + 1) for index in range(len(self.metadata_rows))]
        self.num_periods = len(self.metadata_rows)
        return self

    def __len__(self) -> int:
        return len(self.metadata_rows)

    @staticmethod
    def _array(scalar, shape: tuple[int, ...]) -> torch.Tensor:
        values = scalar.values.to_numpy(zero_copy_only=False)
        return torch.from_numpy(np.asarray(values, dtype=np.float32).reshape(shape).copy())

    def _sample(self, index: int) -> dict[str, torch.Tensor]:
        if self.table is None:
            raise RuntimeError("call build() before indexing")
        row = self.table.slice(index, 1)
        shapes = self.manifest["tensor_shapes"]
        waveform = self._array(row["bp_waveform"][0], tuple(shapes["bp_waveform"]))
        bp = waveform if self._output_value == "waveform" else torch.stack((waveform.min(), waveform.max()))
        if self.manifest["schema"] == "pvi-gcnm-coordinate-direct-parquet-v1":
            if self.channel_mode != "3ch":
                raise ValueError("coordinate-direct v1 supports only [S1,S2,dS2/dt]")
            pvi_hp = self._array(row["s1"][0], tuple(shapes["s1"]))
            pvi_lp = self._array(row["s2"][0], tuple(shapes["s2"]))
        else:
            hp_s1 = self._array(row["hp_s1"][0], tuple(shapes["hp_s1"]))
            hp_s2 = self._array(row["hp_s2"][0], tuple(shapes["hp_s2"]))
            lp_s1 = self._array(row["lp_s1"][0], tuple(shapes["lp_s1"]))
            lp_s2 = self._array(row["lp_s2"][0], tuple(shapes["lp_s2"]))
            if self.channel_mode == "3ch":
                pvi_hp, pvi_lp = hp_s2, lp_s2
            else:
                pvi_hp = torch.cat((hp_s1, hp_s2), dim=0)
                pvi_lp = torch.cat((lp_s1, lp_s2), dim=0)
        return {
            "bp": bp,
            "pviHP": pvi_hp,
            "pviLP": pvi_lp,
            "stats": self._array(row["stats"][0], tuple(shapes["stats"])),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._sample(index)

    def set_partition(
        self, test_size: float = 0.1, shuffle: bool = True, random_state: int = 42, **kwargs
    ) -> None:
        self._split_params = {
            "test_size": test_size,
            "shuffle": shuffle,
            "random_state": random_state,
            **kwargs,
        }

    def _load_split_manifest(self) -> dict:
        source = self._split_manifest_input
        if isinstance(source, dict):
            return source
        if source is not None:
            return json.loads(Path(source).read_text(encoding="utf-8"))
        default = self.root / "splits" / "subject_splits_v1.json"
        if default.is_file():
            return json.loads(default.read_text(encoding="utf-8"))
        return deterministic_subject_split(
            self.metadata_rows,
            test_size=float(self._split_params.get("test_size", 0.1)),
            seed=int(self._split_params.get("random_state", 42)),
        )

    def get_partition(self) -> dict[str, Dataset]:
        manifest = self._load_split_manifest()
        assignments = manifest["assignments"]
        validate_split(self.metadata_rows, assignments)
        train = [i for i, row in enumerate(self.metadata_rows) if assignments[row["sample_id"]] == "train"]
        test = [i for i, row in enumerate(self.metadata_rows) if assignments[row["sample_id"]] == "test"]
        if not train or not test:
            raise ValueError("both train and test partitions must be non-empty")
        self.train_mask = [(i, i + 1) for i in train]
        self.test_mask = [(i, i + 1) for i in test]
        self.subsets = {"train": _ParquetSplit(self, train), "test": _ParquetSplit(self, test)}
        return self.subsets

    def set_dataloaders(self, batch_size: int = 32, shuffle: bool = True, **kwargs) -> None:
        allowed = {
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "drop_last",
            "generator",
        }
        self._loader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            **{key: value for key, value in kwargs.items() if key in allowed},
        }

    def get_dataloaders(self) -> dict[str, DataLoader]:
        if not self.subsets:
            self.get_partition()
        params = dict(self._loader_params or {"batch_size": 32, "shuffle": True})
        train_params = dict(params)
        test_params = dict(params)
        test_params["shuffle"] = False
        self.loaders = {
            "train": DataLoader(self.subsets["train"], **train_params),
            "test": DataLoader(self.subsets["test"], **test_params),
        }
        return self.loaders

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = self.manifest["tensor_shapes"]
        output = (50,) if self._output_value == "waveform" else (2,)
        base = tuple(
            shapes["s1"]
            if self.manifest.get("schema") == "pvi-gcnm-coordinate-direct-parquet-v1"
            else shapes["hp_s2"]
        )
        input_shape = ((1 if self.channel_mode == "3ch" else 2), *base[1:])
        return {
            "input": input_shape,
            "output": output,
            "stats": tuple(shapes["stats"]),
        }

    @property
    def configs(self) -> dict[str, str]:
        return {
            "input_mode": "img",
            "output_mode": self._output_value,
            "mask_key": "mask05",
            "channel_mode": self.channel_mode,
        }

    def state_dict(self) -> dict:
        assigned = {
            self.metadata_rows[i]["sample_id"]
            for i, _ in self.train_mask + self.test_mask
        }
        return {
            "train_sample_ids": [self.metadata_rows[i]["sample_id"] for i, _ in self.train_mask],
            "test_sample_ids": [self.metadata_rows[i]["sample_id"] for i, _ in self.test_mask],
            "excluded_sample_ids": [
                row["sample_id"] for row in self.metadata_rows
                if row["sample_id"] not in assigned
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        train_ids = set(state.get("train_sample_ids", []))
        test_ids = set(state.get("test_sample_ids", []))
        excluded_ids = set(state.get("excluded_sample_ids", []))
        if not train_ids and not test_ids:
            # pvi_ml's TrainingCheckpoint validates components with an empty
            # state roundtrip before the workflow creates its first partition.
            self.train_mask = []
            self.test_mask = []
            self.subsets = {}
            return
        if not train_ids or not test_ids:
            raise ValueError("Parquet checkpoints require non-empty train and test sample IDs")
        overlap = train_ids & test_ids
        if overlap:
            raise ValueError(f"checkpoint assigns {len(overlap)} sample IDs to both partitions")
        assignments = {sample_id: "train" for sample_id in train_ids}
        assignments.update({sample_id: "test" for sample_id in test_ids})
        assignments.update({sample_id: "excluded" for sample_id in excluded_ids})
        self._split_manifest_input = {"assignments": assignments}
        self.get_partition()

    def to(self, device=None, dtype=None, **kwargs):
        return self

    def cleanup(self, attrs=None, placeholder=None) -> None:
        return None

    def print_info(self) -> None:
        print(
            f"{self.name}: {len(self):,} mask05 samples, shapes={self.shapes}, "
            f"train/test={len(self.train_mask):,}/{len(self.test_mask):,}"
        )

    def get_raw_statistics(self) -> dict:
        return {"sqi": None, "num_periods": self.num_periods, "num_seq05": len(self)}

    def get_params_shallow(self) -> dict:
        return {
            "class": type(self).__name__,
            "name": self.name,
            "constituents": sorted({row["source_name"] for row in self.metadata_rows}),
            "configs": self.configs,
            "counts": {
                "num_periods": self.num_periods,
                "num_sequences": len(self),
                "num_train": len(self.train_mask),
                "num_test": len(self.test_mask),
            },
            "shapes": self.shapes,
            "raw_stats": self.get_raw_statistics(),
        }
