"""Pair coordinate-direct Parquet with archived Newton images in source HDF5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from gcnm_pvi.pvi_parquet_dataset import PviParquetCompositeDataset
from gcnm_pvi.pvi_splits import stable_sample_id, validate_split


PERIOD_LENGTH = 50
WINDOW_PERIODS = 5


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _included_records(registry_path: Path, subject: str | None = None) -> list[dict]:
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    records = [
        row
        for row in registry["records"]
        if not row.get("exclusion_reason")
        and (subject is None or str(row["subject"]).lower() == subject.lower())
    ]
    records.sort(key=lambda row: int(row["source_order"]))
    return records


def _zero_based_masks(handle: h5py.File) -> np.ndarray:
    masks = np.asarray(handle["masks/mask05"], dtype=np.int64)
    if masks.ndim == 1:
        masks = masks[None, :]
    masks = masks.copy()
    masks[:, 0] -= 1
    if masks.shape[1] != 2 or np.any(masks[:, 1] - masks[:, 0] != WINDOW_PERIODS):
        raise ValueError("source HDF5 has a non-mask05 window")
    return masks


def read_newton_hdf5_window(
    source_hdf5: str | Path, mask_start: int, mask_stop: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the exact archived Newton HP/LP and target BP window from HDF5."""

    frame_start = int(mask_start) * PERIOD_LENGTH
    frame_stop = int(mask_stop) * PERIOD_LENGTH
    if int(mask_stop) - int(mask_start) != WINDOW_PERIODS:
        raise ValueError("Newton image window must contain five periods")
    with h5py.File(source_hdf5, "r") as handle:
        hp = np.asarray(
            handle["data/pviHP/img"][:, :, frame_start:frame_stop],
            dtype=np.float32,
        )
        lp = np.asarray(
            handle["data/pviLP/img"][:, :, frame_start:frame_stop],
            dtype=np.float32,
        )
        target_start = (int(mask_stop) - 1) * PERIOD_LENGTH
        bp = np.asarray(
            handle["data/bp/signal"][0, target_start:frame_stop],
            dtype=np.float32,
        )
    expected = (40, 40, WINDOW_PERIODS * PERIOD_LENGTH)
    if hp.shape != expected or lp.shape != expected or bp.shape != (PERIOD_LENGTH,):
        raise ValueError(
            f"invalid HDF5 Newton window shapes: hp={hp.shape}, lp={lp.shape}, bp={bp.shape}"
        )
    return hp, lp, bp


class PviNewtonCoordinateHdf5Dataset(PviParquetCompositeDataset):
    """Six-channel dataset without a duplicated Newton-image Parquet root.

    Coordinate S1/S2 and all target metadata remain in immutable Parquet.
    Newton HP/LP frames are loaded once per subject directly from the original
    HDF5 sessions and shared copy-on-write with persistent DataLoader workers.
    """

    def __init__(
        self,
        registry_path: str | Path,
        coordinate_root: str | Path,
        subject: str,
        output_mode: str,
        split_manifest: str | Path | dict,
        name: str | None = None,
    ) -> None:
        super().__init__(
            coordinate_root,
            subjects=[subject],
            input_mode="image",
            output_mode=output_mode,
            mask_key="mask05",
            channel_mode="3ch",
            split_manifest=split_manifest,
            name=name or subject,
        )
        self.registry_path = Path(registry_path)
        self.subject = subject.lower()
        self.coordinate_manifest: dict = {}
        self.source_records: dict[str, dict] = {}
        self.newton_frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def build(self, cleanup: bool = True) -> "PviNewtonCoordinateHdf5Dataset":
        super().build(cleanup=cleanup)
        self.coordinate_manifest = dict(self.manifest)
        records = _included_records(self.registry_path, self.subject)
        if not records:
            raise ValueError(f"registry contains no sessions for {self.subject}")
        self.source_records = {str(row["source_name"]): row for row in records}
        expected_sources = {row["source_name"] for row in self.metadata_rows}
        if set(self.source_records) != expected_sources:
            raise ValueError(
                "coordinate Parquet and HDF5 registry session membership differs"
            )

        rows_by_source: dict[str, list[dict]] = {}
        for row in self.metadata_rows:
            rows_by_source.setdefault(str(row["source_name"]), []).append(row)

        for source_name, record in self.source_records.items():
            source = Path(record["source_hdf5"])
            if not source.is_file():
                raise FileNotFoundError(source)
            with h5py.File(source, "r") as handle:
                masks = _zero_based_masks(handle)
                expected = [
                    (
                        int(start),
                        int(stop),
                        stable_sample_id(source_name, "mask05", int(start), int(stop)),
                    )
                    for start, stop in masks
                ]
                actual = [
                    (
                        int(row["mask_start"]),
                        int(row["mask_stop"]),
                        str(row["sample_id"]),
                    )
                    for row in rows_by_source[source_name]
                ]
                if actual != expected:
                    raise ValueError(
                        f"coordinate/HDF5 mask05 identities differ for {source_name}"
                    )
                hp_node = handle["data/pviHP/img"]
                lp_node = handle["data/pviLP/img"]
                if (
                    hp_node.shape[:2] != (40, 40)
                    or lp_node.shape != hp_node.shape
                    or hp_node.shape[-1] < int(masks[:, 1].max()) * PERIOD_LENGTH
                ):
                    raise ValueError(f"invalid Newton image tensors in {source}")
                # The source is float64 and contiguous. Converting once to
                # float32 exactly matches the former reference-Parquet values
                # and avoids repeated HDF5 reads during thousands of epochs.
                hp = np.asarray(hp_node, dtype=np.float32)
                lp = np.asarray(lp_node, dtype=np.float32)
            self.newton_frames[source_name] = (hp, lp)

        self.manifest = {
            "schema": "pvi-newton-coordinate-hdf5-paired-v1",
            "mask_key": "mask05",
            "tensor_shapes": {
                "pviHP": [2, 40, 40, 250],
                "pviLP": [2, 40, 40, 250],
                "bp_waveform": [50],
                "stats": [2, 5],
            },
            "bp_channel_contract": [
                "newton_hp",
                "d_newton_lp_dt",
                "d2_newton_lp_dt2",
                "s1",
                "s2",
                "d_s2_dt",
            ],
            "coordinate_schema": self.coordinate_manifest["schema"],
            "reference_storage": "source_hdf5",
            "registry": str(self.registry_path.resolve()),
            "registry_sha256": _sha256(self.registry_path),
        }
        return self

    def _sample(self, index: int) -> dict[str, torch.Tensor]:
        if self.table is None or not self.newton_frames:
            raise RuntimeError("call build() before indexing")
        row = self.table.slice(index, 1)
        metadata = self.metadata_rows[index]
        shapes = self.coordinate_manifest["tensor_shapes"]
        waveform = self._array(
            row["bp_waveform"][0], tuple(shapes["bp_waveform"])
        )
        bp = (
            waveform
            if self._output_value == "waveform"
            else torch.stack((waveform.min(), waveform.max()))
        )
        s1 = self._array(row["s1"][0], tuple(shapes["s1"]))
        s2 = self._array(row["s2"][0], tuple(shapes["s2"]))
        frame_start = int(metadata["mask_start"]) * PERIOD_LENGTH
        frame_stop = int(metadata["mask_stop"]) * PERIOD_LENGTH
        hp_frames, lp_frames = self.newton_frames[str(metadata["source_name"])]
        newton_hp = torch.from_numpy(
            hp_frames[:, :, frame_start:frame_stop].copy()
        ).unsqueeze(0)
        newton_lp = torch.from_numpy(
            lp_frames[:, :, frame_start:frame_stop].copy()
        ).unsqueeze(0)
        return {
            "bp": bp,
            "pviHP": torch.cat((newton_hp, s1), dim=0),
            "pviLP": torch.cat((newton_lp, s2), dim=0),
            "stats": self._array(row["stats"][0], tuple(shapes["stats"])),
        }

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "input": (2, 40, 40, 250),
            "output": (50,) if self._output_value == "waveform" else (2,),
            "stats": (2, 5),
        }

    @property
    def configs(self) -> dict[str, str]:
        return {
            "input_mode": "img",
            "output_mode": self._output_value,
            "mask_key": "mask05",
            "channel_contract": "newton_hdf5_3_plus_coordinate_parquet_3",
        }

    def print_info(self) -> None:
        cached = sum(
            hp.nbytes + lp.nbytes for hp, lp in self.newton_frames.values()
        )
        print(
            f"{self.name}: {len(self):,} HDF5-Newton+coordinate samples, "
            f"cache={cached / 2**30:.2f} GiB, shapes={self.shapes}, "
            f"train/test={len(self.train_mask):,}/{len(self.test_mask):,}"
        )


def validate_coordinate_hdf5_contract(
    coordinate_root: Path, registry_path: Path, split_path: Path
) -> dict:
    """Validate every stable identity and target against source HDF5."""

    import pyarrow.dataset as pads

    coordinate_root = Path(coordinate_root)
    if (coordinate_root / "_INCOMPLETE").exists():
        raise RuntimeError(f"incomplete coordinate root: {coordinate_root}")
    manifest = json.loads(
        (coordinate_root / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") != "pvi-gcnm-coordinate-direct-parquet-v1":
        raise ValueError("expected coordinate-direct Parquet")
    columns = [
        "sample_id",
        "subject",
        "session",
        "source_name",
        "source_order",
        "mask_start",
        "mask_stop",
        "num_periods",
        "bp_waveform",
        "stats",
    ]
    table = pads.dataset(
        str(coordinate_root / "shards"), format="parquet"
    ).to_table(columns=columns).sort_by(
        [
            ("source_order", "ascending"),
            ("mask_start", "ascending"),
            ("mask_stop", "ascending"),
        ]
    )
    records = _included_records(registry_path)
    expected_rows: list[dict] = []
    offset = 0
    for record in records:
        source = Path(record["source_hdf5"])
        with h5py.File(source, "r") as handle:
            masks = _zero_based_masks(handle)
            source_num_periods = int(
                np.asarray(handle["metadata/num_periods"]).item()
            )
            count = len(masks)
            session_table = table.slice(offset, count)
            source_names = session_table["source_name"].to_pylist()
            if source_names != [record["source_name"]] * count:
                raise ValueError(
                    f"coordinate ordering differs at {record['source_name']}"
                )
            starts = masks[:, 0]
            stops = masks[:, 1]
            bp_indices = (
                (stops - 1)[:, None] * PERIOD_LENGTH
                + np.arange(PERIOD_LENGTH)[None, :]
            )
            expected_bp = np.asarray(
                handle["data/bp/signal"][0], dtype=np.float32
            )[bp_indices]
            period_indices = starts[:, None] + np.arange(WINDOW_PERIODS)[None, :]
            expected_stats = np.stack(
                (
                    np.asarray(
                        handle["stats/pviHP/duration"][0], dtype=np.float32
                    )[period_indices],
                    np.asarray(
                        handle["stats/pviHP/tMax"][0], dtype=np.float32
                    )[period_indices],
                ),
                axis=1,
            )
            actual_bp = np.asarray(
                session_table["bp_waveform"].combine_chunks().values
            ).reshape(count, PERIOD_LENGTH)
            actual_stats = np.asarray(
                session_table["stats"].combine_chunks().values
            ).reshape(count, 2, WINDOW_PERIODS)
            if not np.array_equal(actual_bp, expected_bp, equal_nan=True):
                raise ValueError(f"BP differs from HDF5 for {record['source_name']}")
            if not np.array_equal(actual_stats, expected_stats, equal_nan=True):
                raise ValueError(
                    f"stats differ from HDF5 for {record['source_name']}"
                )
            for start, stop in masks:
                expected_rows.append(
                    {
                        "sample_id": stable_sample_id(
                            record["source_name"],
                            "mask05",
                            int(start),
                            int(stop),
                        ),
                        "subject": record["subject"],
                        "session": record["session"],
                        "source_name": record["source_name"],
                        "source_order": int(record["source_order"]),
                        "mask_start": int(start),
                        "mask_stop": int(stop),
                        # This column is the full source-session period count
                        # used by pvi_ml's composite-session offsets. The
                        # individual mask window length is stop-start == 5.
                        "num_periods": source_num_periods,
                    }
                )
            offset += count
    if offset != table.num_rows:
        raise ValueError(
            f"HDF5/coordinate row count differs: {offset} != {table.num_rows}"
        )
    actual_rows = table.select(columns[:8]).to_pylist()
    if actual_rows != expected_rows:
        raise ValueError("coordinate sample identities differ from source HDF5")
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if set(split["assignments"]) != {row["sample_id"] for row in actual_rows}:
        raise ValueError("frozen split and HDF5-backed sample IDs differ")
    validate_split(actual_rows, split["assignments"])
    subjects = {row["subject"] for row in actual_rows}
    return {
        "status": "pass",
        "rows": len(actual_rows),
        "subjects": len(subjects),
        "sessions": len(records),
        "newton_storage": "source_hdf5",
    }
