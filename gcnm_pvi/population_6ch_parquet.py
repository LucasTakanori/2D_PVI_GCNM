"""Fast population-wide six-channel dataset backed by row-grouped Parquet."""

from __future__ import annotations

import bisect
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class ModeValue(str):
    @property
    def value(self) -> str:
        return str(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RowGroupBatchSampler(Sampler[list[int]]):
    """Shuffle row groups while retaining contiguous Parquet reads per batch."""

    def __init__(
        self,
        groups: list[tuple[int, int]],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.batches: list[list[int]] = []
        for start, stop in groups:
            for first in range(start, stop, batch_size):
                self.batches.append(list(range(first, min(first + batch_size, stop))))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
            self.epoch += 1
        # Yielding makes this method lazy. PyTorch's multiprocessing iterator
        # requests an iterator twice during initial worker startup and discards
        # the first without consuming it; sampler state must advance only when
        # a batch is actually requested.
        for index in order:
            yield self.batches[index]


class PrecomputedPviBatchSampler(Sampler[list[int]]):
    """Replay an ahead-of-time PVI sampler schedule exactly once per epoch."""

    def __init__(
        self,
        schedule_path: str | Path,
        *,
        row_count: int,
        batch_size: int,
        start_epoch: int = 0,
    ) -> None:
        self.schedule_path = Path(schedule_path)
        self.schedule = np.load(self.schedule_path, mmap_mode="r")
        if self.schedule.dtype != np.int32 or self.schedule.ndim != 2:
            raise ValueError(f"invalid PVI schedule array: {self.schedule_path}")
        if int(self.schedule.shape[1]) != int(row_count):
            raise ValueError(
                f"schedule row count differs: {self.schedule.shape[1]} != {row_count}"
            )
        self.row_count = int(row_count)
        self.batch_size = int(batch_size)
        self.epoch = int(start_epoch)

    def __len__(self) -> int:
        return (self.row_count + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # This is deliberately a generator rather than a method that returns a
        # pre-built iterator. With num_workers>0, PyTorch calls iter(sampler)
        # twice when workers first start but consumes only the second iterator.
        # Deferring the epoch increment until the first next() keeps one stored
        # PVI schedule aligned with one real training/evaluation pass.
        if self.epoch >= int(self.schedule.shape[0]):
            raise RuntimeError(
                f"precomputed PVI schedules exhausted at epoch {self.epoch}; "
                f"available={self.schedule.shape[0]}"
            )
        order = self.schedule[self.epoch].tolist()
        self.epoch += 1
        for first in range(0, self.row_count, self.batch_size):
            yield order[first : first + self.batch_size]


class PopulationSixChannelSplit(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        *,
        relative_files: list[str] | None = None,
        selection_indices: np.ndarray | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.files = (
            [self.root / relative for relative in relative_files]
            if relative_files is not None
            else sorted((self.root / split).glob("*.parquet"))
        )
        if not self.files:
            raise FileNotFoundError(f"no {split} shards under {self.root}")
        for path in self.files:
            if not path.is_file():
                raise FileNotFoundError(path)
        self._parquet_handles: dict[Path, pq.ParquetFile] = {}
        self.selection_indices = selection_indices
        if self.selection_indices is not None:
            if self.selection_indices.dtype != np.int32 or self.selection_indices.ndim != 1:
                raise ValueError(f"invalid {split} selection index array")
        self.row_groups: list[tuple[Path, int, int, int]] = []
        self.bounds: list[tuple[int, int]] = []
        offset = 0
        for path in self.files:
            metadata = pq.ParquetFile(path).metadata
            for group in range(metadata.num_row_groups):
                count = int(metadata.row_group(group).num_rows)
                self.row_groups.append((path, group, offset, offset + count))
                self.bounds.append((offset, offset + count))
                offset += count
        self.cumulative_stops = [stop for _, stop in self.bounds]
        self.physical_row_count = offset
        self.row_count = (
            len(self.selection_indices)
            if self.selection_indices is not None
            else self.physical_row_count
        )
        if self.selection_indices is not None:
            if len(np.unique(self.selection_indices)) != len(self.selection_indices):
                raise ValueError(f"duplicate physical rows in {split} selection")
            if len(self.selection_indices) and (
                int(self.selection_indices.min()) < 0
                or int(self.selection_indices.max()) >= self.physical_row_count
            ):
                raise ValueError(f"out-of-range physical row in {split} selection")
        self.input_mode = ModeValue("img")
        self.output_mode = ModeValue("waveform")
        self.mask_key = ModeValue("mask05")

    def __len__(self) -> int:
        return self.row_count

    @staticmethod
    def _array(table, name: str, shape: tuple[int, ...]) -> np.ndarray:
        values = table[name].combine_chunks().values.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=np.float32).reshape((table.num_rows, *shape))

    def _group_for_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.row_count
        if index < 0 or index >= self.row_count:
            raise IndexError(index)
        physical_index = (
            int(self.selection_indices[index])
            if self.selection_indices is not None
            else index
        )
        group = bisect.bisect_right(self.cumulative_stops, physical_index)
        return group, physical_index - self.bounds[group][0]

    def metadata_rows(self, columns: list[str]) -> list[dict]:
        """Read projected metadata in this split's logical row order."""

        physical_rows: list[dict] = []
        for path in self.files:
            physical_rows.extend(
                pq.read_table(path, columns=columns, use_threads=False).to_pylist()
            )
        if len(physical_rows) != self.physical_row_count:
            raise RuntimeError(f"metadata row count differs for {self.split}")
        if self.selection_indices is None:
            return physical_rows
        return [physical_rows[int(index)] for index in self.selection_indices]

    def _parquet(self, path: Path) -> pq.ParquetFile:
        parquet = self._parquet_handles.get(path)
        if parquet is None:
            parquet = pq.ParquetFile(path)
            self._parquet_handles[path] = parquet
        return parquet

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_parquet_handles"] = {}
        return state

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]]:
        requests: dict[int, list[tuple[int, int]]] = {}
        for output_position, index in enumerate(indices):
            group, local = self._group_for_index(int(index))
            requests.setdefault(group, []).append((output_position, local))
        output: list[dict[str, torch.Tensor] | None] = [None] * len(indices)
        for group, positions in requests.items():
            path, row_group, _, _ = self.row_groups[group]
            table = self._parquet(path).read_row_group(
                row_group,
                columns=["pviHP", "pviLP", "bp", "stats"],
                use_threads=False,
            )
            hp = self._array(table, "pviHP", (2, 40, 40, 250))
            lp = self._array(table, "pviLP", (2, 40, 40, 250))
            bp = self._array(table, "bp", (50,))
            stats = self._array(table, "stats", (2, 5))
            for output_position, local in positions:
                output[output_position] = {
                    "pviHP": torch.from_numpy(hp[local].copy()),
                    "pviLP": torch.from_numpy(lp[local].copy()),
                    "bp": torch.from_numpy(bp[local].copy()),
                    "stats": torch.from_numpy(stats[local].copy()),
                }
        return output  # type: ignore[return-value]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.__getitems__([index])[0]


class PopulationSixChannelParquetDataset:
    """pvi_ml-compatible population cohort with immutable cached partitions."""

    SUPPORTED_SCHEMAS = {
        "pvi-gcnm-population-6ch-parquet-v1",
        "pvi-gcnm-population-6ch-parquet-v2",
        "pvi-gcnm-population-6ch-parquet-v3",
        "pvi-gcnm-population-6ch-parquet-view-v4",
    }

    def __init__(
        self,
        cache_root: str | Path,
        *,
        seed: int = 42,
        require_prestratified: bool = False,
        require_exact_pvi_schedules: bool = False,
    ) -> None:
        self.root = Path(cache_root)
        # Preserve PVI-ML's dataset name so its artifact filenames remain
        # dataset_lazy_{checkpoints,configs,history,results,statistics}.*.
        self.name = "dataset_lazy"
        self.path = str(self.root)
        self.input_mode = ModeValue("img")
        self.output_mode = ModeValue("waveform")
        self.mask_key = ModeValue("mask05")
        self.seed = int(seed)
        self.require_prestratified = bool(require_prestratified)
        self.require_exact_pvi_schedules = bool(require_exact_pvi_schedules)
        self.prestratified = False
        self.exact_pvi_schedules = False
        self.index_view = False
        self.split_mode = "within"
        self.test_subgroups: list[str] = []
        self.train_subgroups: list[str] = []
        self.manifest: dict = {}
        self.subsets: dict[str, PopulationSixChannelSplit] = {}
        self.loaders: dict[str, DataLoader] = {}
        self.active_mask: list[tuple[int, int]] = []
        self.train_mask: list[tuple[int, int]] = []
        self.test_mask: list[tuple[int, int]] = []
        self._loader_params: dict = {}
        self._split_params: dict = {}
        self.offset_period = 0
        self.period_length = 50
        self.num_periods = 0
        self.raws = []
        self._pending_sampler_epochs = {"train": 0, "test": 0}

    def build(self):
        if (self.root / "_INCOMPLETE").exists() or not (self.root / "_SUCCESS").is_file():
            raise RuntimeError(f"population cache is incomplete: {self.root}")
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") not in self.SUPPORTED_SCHEMAS:
            raise ValueError("population cache has the wrong schema")
        layout = self.manifest.get("batch_layout", {})
        self.prestratified = (
            self.manifest.get("schema") == "pvi-gcnm-population-6ch-parquet-v2"
            and layout.get("strategy") == "pvi-source-clustered-prestratified"
        )
        self.exact_pvi_schedules = (
            self.manifest.get("schema")
            in {
                "pvi-gcnm-population-6ch-parquet-v3",
                "pvi-gcnm-population-6ch-parquet-view-v4",
            }
            and layout.get("strategy") == "precomputed-exact-pvi-batch-schedules"
        )
        self.index_view = (
            self.manifest.get("schema") == "pvi-gcnm-population-6ch-parquet-view-v4"
            and self.manifest.get("data_view", {}).get("strategy")
            == "immutable-physical-row-index"
        )
        self.split_mode = str(self.manifest.get("split_mode", "within"))
        self.test_subgroups = list(map(str, self.manifest.get("test_subjects", [])))
        self.train_subgroups = list(map(str, self.manifest.get("train_subjects", [])))
        if self.require_prestratified and not self.prestratified:
            raise ValueError(
                "training requires a v2 cache with PVI-style pre-stratified row groups"
            )
        if self.require_exact_pvi_schedules and not self.exact_pvi_schedules:
            raise ValueError(
                "training requires a v3 random-access cache with exact PVI schedules"
            )
        if self.exact_pvi_schedules:
            if int(self.manifest.get("row_group_size", -1)) != 1:
                raise ValueError("exact PVI cache must store one sample per Parquet row group")
            for split in ("train", "test"):
                schedule_path = self.root / layout["schedule_files"][split]
                schedule = np.load(schedule_path, mmap_mode="r")
                expected_shape = (
                    int(layout["epochs"]),
                    int(self.manifest["counts"][split]),
                )
                if schedule.dtype != np.int32 or schedule.shape != expected_shape:
                    raise ValueError(
                        f"invalid {split} schedule: {schedule.dtype}/{schedule.shape}, "
                        f"expected int32/{expected_shape}"
                    )
                expected_digest = layout.get("schedule_sha256", {}).get(split)
                if expected_digest and _sha256(schedule_path) != expected_digest:
                    raise ValueError(f"{split} PVI schedule checksum differs from manifest")
        if int(self.manifest.get("seed", -1)) != self.seed:
            raise ValueError("population cache split seed differs from requested seed")
        if self.index_view:
            view = self.manifest["data_view"]
            if view.get("source_caches"):
                source_root = Path("/")
                source_files = list(view["source_files"])
                for source_cache in view["source_caches"]:
                    cache_root = Path(source_cache["root"]).resolve()
                    source_manifest = cache_root / "manifest.json"
                    if not (cache_root / "_SUCCESS").is_file() or not source_manifest.is_file():
                        raise RuntimeError(
                            f"index-view source cache is incomplete: {cache_root}"
                        )
                    if _sha256(source_manifest) != source_cache["manifest_sha256"]:
                        raise ValueError("index-view source cache manifest changed")
            else:
                source_root = Path(view["source_root"]).resolve()
                source_manifest = source_root / "manifest.json"
                if not (source_root / "_SUCCESS").is_file() or not source_manifest.is_file():
                    raise RuntimeError(f"index-view source cache is incomplete: {source_root}")
                if _sha256(source_manifest) != view["source_manifest_sha256"]:
                    raise ValueError("index-view source cache manifest changed")
                source_files = list(view["source_files"])
            selections = {}
            for split in ("train", "test"):
                selection_path = self.root / view["selection_files"][split]
                selection = np.load(selection_path, mmap_mode="r")
                expected_digest = view.get("selection_sha256", {}).get(split)
                if expected_digest and _sha256(selection_path) != expected_digest:
                    raise ValueError(f"{split} selection checksum differs from manifest")
                selections[split] = selection
            if set(selections["train"].tolist()) & set(selections["test"].tolist()):
                raise ValueError("index-view train and test selections overlap")
            self.subsets = {
                split: PopulationSixChannelSplit(
                    source_root,
                    split,
                    relative_files=source_files,
                    selection_indices=selections[split],
                )
                for split in ("train", "test")
            }
            physical_count = self.subsets["train"].physical_row_count
            selected = set(selections["train"].tolist()) | set(selections["test"].tolist())
            if selected != set(range(physical_count)):
                raise ValueError("index-view selections do not partition the source rows")
        else:
            self.subsets = {
                split: PopulationSixChannelSplit(
                    self.root,
                    split,
                    relative_files=self.manifest.get("files", {}).get(split),
                )
                for split in ("train", "test")
            }
        n_train, n_test = len(self.subsets["train"]), len(self.subsets["test"])
        expected = self.manifest["counts"]
        if (n_train, n_test) != (int(expected["train"]), int(expected["test"])):
            raise ValueError("Parquet row counts differ from manifest")
        self.train_mask = [(index, index + 1) for index in range(n_train)]
        self.test_mask = [(n_train + index, n_train + index + 1) for index in range(n_test)]
        self.active_mask = self.train_mask + self.test_mask
        self.num_periods = n_train + n_test
        return self

    def __len__(self) -> int:
        return len(self.active_mask)

    def set_partition(self, test_size=0.1, shuffle=True, split_mode="within", random_state=42, **kwargs):
        requested_mode = str(split_mode)
        accepted_modes = {"within", "local"} if self.split_mode == "within" else {self.split_mode}
        if float(test_size) != 0.1 or requested_mode not in accepted_modes:
            raise ValueError(
                f"cache is frozen to population-{self.split_mode} test_size=0.1"
            )
        if int(random_state) != self.seed:
            raise ValueError("requested split seed differs from frozen cache")
        self._split_params = {
            "test_size": 0.1,
            "shuffle": bool(shuffle),
            "split_mode": self.split_mode,
            "random_state": self.seed,
        }

    def get_partition(self):
        if not self.subsets:
            self.build()
        return self.subsets

    def set_dataloaders(self, batch_size=64, shuffle=True, num_workers=8, **kwargs):
        if self.prestratified or self.exact_pvi_schedules:
            layout_batch_size = int(self.manifest["batch_layout"]["batch_size"])
            if int(batch_size) != layout_batch_size:
                raise ValueError(
                    "the materialized Parquet/schedule contract fixes batch size: "
                    f"requested batch_size={batch_size}, cache batch_size={layout_batch_size}"
                )
        self._loader_params = {
            "batch_size": int(batch_size),
            "shuffle": False if self.exact_pvi_schedules else bool(shuffle),
            "num_workers": int(num_workers),
            **kwargs,
        }
        if self.exact_pvi_schedules:
            self._loader_params.update(
                {
                    "stratified": True,
                    "cluster_size": int(self.manifest["batch_layout"]["cluster_size"]),
                }
            )

    def get_dataloaders(self):
        params = dict(self._loader_params or {"batch_size": 64, "shuffle": True, "num_workers": 8})
        batch_size = int(params.pop("batch_size"))
        shuffle = bool(params.pop("shuffle", True))
        num_workers = int(params.pop("num_workers", 0))
        worker_options = {
            key: params[key]
            for key in ("pin_memory", "persistent_workers", "prefetch_factor")
            if key in params and not (key in {"persistent_workers", "prefetch_factor"} and num_workers == 0)
        }
        self.loaders = {}
        for split in ("train", "test"):
            subset = self.subsets[split]
            if self.exact_pvi_schedules:
                schedule_path = self.root / self.manifest["batch_layout"]["schedule_files"][split]
                sampler = PrecomputedPviBatchSampler(
                    schedule_path,
                    row_count=len(subset),
                    batch_size=batch_size,
                    start_epoch=self._pending_sampler_epochs[split],
                )
            else:
                sampler = RowGroupBatchSampler(
                    subset.bounds,
                    batch_size,
                    shuffle=shuffle if split == "train" else False,
                    seed=self.seed,
                )
            self.loaders[split] = DataLoader(
                subset,
                batch_sampler=sampler,
                num_workers=num_workers,
                **worker_options,
            )
        return self.loaders

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {"input": (2, 40, 40, 250), "output": (50,), "stats": (2, 5)}

    @property
    def configs(self) -> dict[str, str]:
        return {
            "input_mode": "img",
            "output_mode": "waveform",
            "mask_key": "mask05",
            "channel_contract": "newton3_plus_gcnm3",
            "split_mode": self.split_mode,
            "batch_layout": (
                "precomputed-exact-pvi-batch-schedules"
                if self.exact_pvi_schedules
                else (
                    "pvi-source-clustered-prestratified"
                    if self.prestratified
                    else "source-ordered"
                )
            ),
        }

    def state_dict(self) -> dict:
        sampler_epochs = dict(self._pending_sampler_epochs)
        for split, loader in self.loaders.items():
            epoch = getattr(loader.batch_sampler, "epoch", None)
            if epoch is not None:
                sampler_epochs[split] = int(epoch)
        return {
            "cache_root": str(self.root.resolve()),
            "manifest_sha256": _sha256(self.root / "manifest.json"),
            "seed": self.seed,
            "active_mask": self.active_mask,
            "train_mask": self.train_mask,
            "test_mask": self.test_mask,
            "test_subgroups": sorted(self.test_subgroups),
            "train_subgroups": sorted(self.train_subgroups),
            "offset_period": self.offset_period,
            "sampler_epochs": sampler_epochs,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("cache_root") and Path(state["cache_root"]).resolve() != self.root.resolve():
            raise ValueError("checkpoint references a different population cache")
        if state.get("manifest_sha256") and state["manifest_sha256"] != _sha256(self.root / "manifest.json"):
            raise ValueError("population cache manifest changed since checkpoint")
        if state.get("seed") is not None and int(state["seed"]) != self.seed:
            raise ValueError("checkpoint split seed differs")
        for key, expected in (
            ("test_subgroups", self.test_subgroups),
            ("train_subgroups", self.train_subgroups),
        ):
            if state.get(key) is not None and sorted(map(str, state[key])) != sorted(expected):
                raise ValueError(f"checkpoint {key} differ from the frozen cache")
        if state.get("sampler_epochs"):
            self._pending_sampler_epochs = {
                split: int(state["sampler_epochs"].get(split, 0))
                for split in ("train", "test")
            }

    def cleanup(self, attrs=None, placeholder=None) -> None:
        return None

    def to(self, device=None, dtype=None, **kwargs):
        return self

    def get_raw_statistics(self) -> dict:
        return {
            "sqi": None,
            "num_periods": self.num_periods,
            "num_seq01": 0,
            "num_seq05": self.num_periods,
            "num_seq10": 0,
            "num_seq15": 0,
        }

    def get_params_shallow(self) -> dict:
        return {
            "class": type(self).__name__,
            "name": self.name,
            "constituents": [self.root.name],
            "configs": self.configs,
            "counts": {
                "num_periods": self.num_periods,
                "num_sequences": len(self),
                "num_train": len(self.train_mask),
                "num_test": len(self.test_mask),
            },
            "cache_root": str(self.root.resolve()),
            "cache_manifest_sha256": _sha256(self.root / "manifest.json"),
            "test_subgroups": self.test_subgroups,
            "train_subgroups": self.train_subgroups,
            "split_params": self._split_params,
            "batch_params": self._loader_params,
            "shapes": self.shapes,
            "raw_stats": self.get_raw_statistics(),
        }

    def print_info(self) -> None:
        print(
            f"{self.name}: {len(self):,} population-{self.split_mode} samples, "
            f"train/test={len(self.train_mask):,}/{len(self.test_mask):,}, "
            f"shapes={self.shapes}"
        )
