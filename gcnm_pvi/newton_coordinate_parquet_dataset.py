"""Paired archived-Newton and coordinate-direct Parquet dataset."""

from __future__ import annotations

from pathlib import Path

import torch

from gcnm_pvi.pvi_parquet_dataset import PviParquetCompositeDataset
from gcnm_pvi.reference_parquet_dataset import PviReferenceParquetDataset


class PviNewtonCoordinateCombinedDataset(PviReferenceParquetDataset):
    """Join two immutable roots by stable sample ID without duplicating tensors.

    ``pviHP`` carries ``[Newton HP, S1]`` and ``pviLP`` carries
    ``[Newton LP, S2]``. The BP model preprocessor expands these four stored
    fields into the selected six-channel tensor.
    """

    def __init__(
        self,
        reference_root: str | Path,
        coordinate_root: str | Path,
        subject: str,
        output_mode: str,
        split_manifest: str | Path | dict,
        name: str | None = None,
    ) -> None:
        super().__init__(
            reference_root, subject, output_mode, split_manifest, name=name
        )
        self.coordinate_root = Path(coordinate_root)
        self.coordinate: PviParquetCompositeDataset | None = None
        self.reference_manifest: dict = {}
        self.coordinate_manifest: dict = {}

    def build(self, cleanup: bool = True) -> "PviNewtonCoordinateCombinedDataset":
        super().build(cleanup=cleanup)
        if self.manifest.get("input_mode") != "img":
            raise ValueError("six-channel fusion requires archived Newton images")
        self.reference_manifest = dict(self.manifest)
        coordinate = PviParquetCompositeDataset(
            self.coordinate_root,
            subjects=[self.subject],
            output_mode=self._output_value,
            channel_mode="3ch",
            split_manifest=self._split_manifest_input,
            name=self.name,
            verbose=False,
        ).build()
        identity_fields = (
            "sample_id", "subject", "session", "source_name", "mask_start", "mask_stop"
        )
        reference_identities = [
            tuple(row[field] for field in identity_fields) for row in self.metadata_rows
        ]
        coordinate_identities = [
            tuple(row[field] for field in identity_fields)
            for row in coordinate.metadata_rows
        ]
        if reference_identities != coordinate_identities:
            raise ValueError("Newton and coordinate Parquet sample ordering differs")
        for column in ("bp_waveform", "stats"):
            if not self.table[column].equals(coordinate.table[column]):
                raise ValueError(f"Newton and coordinate {column} values differ")
        self.coordinate = coordinate
        self.coordinate_manifest = dict(coordinate.manifest)
        self.manifest = {
            "schema": "pvi-newton-coordinate-paired-v1",
            "mask_key": "mask05",
            "tensor_shapes": self.reference_manifest["tensor_shapes"],
            "bp_channel_contract": [
                "newton_hp",
                "d_newton_lp_dt",
                "d2_newton_lp_dt2",
                "s1",
                "s2",
                "d_s2_dt",
            ],
            "reference_schema": self.reference_manifest["schema"],
            "coordinate_schema": self.coordinate_manifest["schema"],
        }
        return self

    def _sample(self, index: int) -> dict[str, torch.Tensor]:
        if self.coordinate is None:
            raise RuntimeError("call build() before indexing")
        reference = super()._sample(index)
        coordinate = self.coordinate._sample(index)
        if not torch.equal(reference["bp"], coordinate["bp"]):
            raise ValueError("paired BP values differ")
        if not torch.equal(reference["stats"], coordinate["stats"]):
            raise ValueError("paired stats differ")
        return {
            "bp": reference["bp"],
            "pviHP": torch.cat((reference["pviHP"], coordinate["pviHP"]), dim=0),
            "pviLP": torch.cat((reference["pviLP"], coordinate["pviLP"]), dim=0),
            "stats": reference["stats"],
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
            "channel_contract": "newton3_plus_coordinate3",
        }

    def print_info(self) -> None:
        print(
            f"{self.name}: {len(self):,} paired Newton+coordinate samples, "
            f"shapes={self.shapes}, train/test={len(self.train_mask):,}/{len(self.test_mask):,}"
        )
