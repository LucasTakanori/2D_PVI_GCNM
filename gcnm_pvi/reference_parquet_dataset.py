"""Drop-in pvi_ml dataset for archived reference-image and BioZ Parquet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from gcnm_pvi.pvi_splits import validate_split


class _ModeValue(str):
    @property
    def value(self) -> str:
        return str(self)


class _Split(Dataset):
    def __init__(self, owner: "PviReferenceParquetDataset", indices: list[int]) -> None:
        self.owner = owner
        self.indices = indices
        self.input_mode = owner.input_mode
        self.output_mode = owner.output_mode
        self.mask_key = owner.mask_key

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.owner._sample(self.indices[index])


class PviReferenceParquetDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        subject: str,
        output_mode: str,
        split_manifest: str | Path | dict,
        name: str | None = None,
    ) -> None:
        output = str(output_mode).lower()
        if output not in {"waveform", "fiducials"}:
            raise ValueError("output_mode must be waveform or fiducials")
        self.root = Path(root)
        self.subject = subject.lower()
        self.name = name or self.subject
        self.path = str(self.root)
        self._output_value = output
        self.output_mode = _ModeValue(output)
        self.mask_key = _ModeValue("mask05")
        self._split_manifest_input = split_manifest
        self.input_mode = _ModeValue("img")
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
        self.raws = []
        self.period_length = 50
        self.num_periods = 0

    def build(self, cleanup: bool = True) -> "PviReferenceParquetDataset":
        import pyarrow.dataset as pads

        if (self.root / "_INCOMPLETE").exists():
            raise RuntimeError(f"reference export is incomplete: {self.root}")
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != "pvi-reference-parquet-v1":
            raise ValueError("unsupported reference Parquet schema")
        mode = self.manifest.get("input_mode")
        if mode not in {"img", "bioz"}:
            raise ValueError(f"unsupported reference input mode {mode!r}")
        self.input_mode = _ModeValue("img" if mode == "img" else "impedance")
        dataset = pads.dataset(str(self.root / "shards"), format="parquet")
        self.table = dataset.to_table(filter=pads.field("subject") == self.subject).sort_by(
            [("source_order", "ascending"), ("mask_start", "ascending"), ("mask_stop", "ascending")]
        )
        self.metadata_rows = self.table.select(
            ["sample_id", "subject", "session", "source_name", "mask_start", "mask_stop", "num_periods"]
        ).to_pylist()
        if not self.metadata_rows:
            raise ValueError(f"subject {self.subject} selected zero rows")
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
        return {
            "bp": bp,
            "pviHP": self._array(row["pviHP"][0], tuple(shapes["pviHP"])),
            "pviLP": self._array(row["pviLP"][0], tuple(shapes["pviLP"])),
            "stats": self._array(row["stats"][0], tuple(shapes["stats"])),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._sample(index)

    def set_partition(self, test_size: float = 0.1, shuffle: bool = True, random_state: int = 42, **kwargs) -> None:
        self._split_params = {"test_size": test_size, "shuffle": shuffle, "random_state": random_state, **kwargs}

    def _split_manifest(self) -> dict:
        source = self._split_manifest_input
        if isinstance(source, dict):
            return source
        return json.loads(Path(source).read_text(encoding="utf-8"))

    def get_partition(self) -> dict[str, Dataset]:
        assignments = self._split_manifest()["assignments"]
        validate_split(self.metadata_rows, assignments)
        train = [i for i, row in enumerate(self.metadata_rows) if assignments[row["sample_id"]] == "train"]
        test = [i for i, row in enumerate(self.metadata_rows) if assignments[row["sample_id"]] == "test"]
        if not train or not test:
            raise ValueError("both train and test partitions must be non-empty")
        self.train_mask = [(i, i + 1) for i in train]
        self.test_mask = [(i, i + 1) for i in test]
        self.subsets = {"train": _Split(self, train), "test": _Split(self, test)}
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
        self._loader_params = {"batch_size": batch_size, "shuffle": shuffle, **{k: v for k, v in kwargs.items() if k in allowed}}

    def get_dataloaders(self) -> dict[str, DataLoader]:
        if not self.subsets:
            self.get_partition()
        train_params = dict(self._loader_params or {"batch_size": 32, "shuffle": True})
        test_params = dict(train_params)
        test_params["shuffle"] = False
        self.loaders = {
            "train": DataLoader(self.subsets["train"], **train_params),
            "test": DataLoader(self.subsets["test"], **test_params),
        }
        return self.loaders

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = self.manifest["tensor_shapes"]
        return {
            "input": tuple(shapes["pviHP"]),
            "output": (50,) if self._output_value == "waveform" else (2,),
            "stats": tuple(shapes["stats"]),
        }

    @property
    def configs(self) -> dict[str, str]:
        return {"input_mode": self.input_mode.value, "output_mode": self.output_mode.value, "mask_key": "mask05"}

    def state_dict(self) -> dict:
        assigned = {self.metadata_rows[i]["sample_id"] for i, _ in self.train_mask + self.test_mask}
        return {
            "train_sample_ids": [self.metadata_rows[i]["sample_id"] for i, _ in self.train_mask],
            "test_sample_ids": [self.metadata_rows[i]["sample_id"] for i, _ in self.test_mask],
            "excluded_sample_ids": [row["sample_id"] for row in self.metadata_rows if row["sample_id"] not in assigned],
        }

    def load_state_dict(self, state: dict) -> None:
        train_ids = set(state.get("train_sample_ids", []))
        test_ids = set(state.get("test_sample_ids", []))
        excluded_ids = set(state.get("excluded_sample_ids", []))
        if not train_ids and not test_ids:
            self.train_mask, self.test_mask, self.subsets = [], [], {}
            return
        if not train_ids or not test_ids or train_ids & test_ids:
            raise ValueError("invalid Parquet checkpoint split state")
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
        print(f"{self.name}: {len(self):,} reference Parquet samples, shapes={self.shapes}, train/test={len(self.train_mask):,}/{len(self.test_mask):,}")

    def get_raw_statistics(self) -> dict:
        return {"sqi": None, "num_periods": self.num_periods, "num_seq05": len(self)}

    def get_params_shallow(self) -> dict:
        return {
            "class": type(self).__name__, "name": self.name,
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

    def _validate_components(self, components) -> None:
        return None
