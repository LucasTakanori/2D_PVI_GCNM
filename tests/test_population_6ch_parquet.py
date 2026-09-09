import hashlib
import json
import random
from functools import reduce
from operator import concat

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from gcnm_pvi.population_6ch_parquet import (
    PopulationSixChannelParquetDataset,
    PrecomputedPviBatchSampler,
    RowGroupBatchSampler,
)
from scripts.build_population_6ch_exact_pvi_cache import (
    build_exact_cache,
    pvi_stratified_order,
    require_scratch_path,
)
from scripts.build_population_6ch_disjoint_view import build_disjoint_view
from scripts.build_population_disjoint_split import build_disjoint_manifest
from scripts.build_population_6ch_cache import process_shard, sample_ids_from_cache
from scripts.repack_population_6ch_stratified import repack_cache
from gcnm_pvi.train_population_6ch_bp import experiment_target, six_channel_process_sequence


def _fixed(values):
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32()), values.shape[1]
    )


def _write_split(root, split, count, *, row_group_size=2):
    width = 2 * 40 * 40 * 250
    hp = np.arange(count * width, dtype=np.float32).reshape(count, width)
    lp = -hp
    bp = np.arange(count * 50, dtype=np.float32).reshape(count, 50)
    stats = np.arange(count * 10, dtype=np.float32).reshape(count, 10)
    table = pa.Table.from_arrays(
        [_fixed(hp), _fixed(lp), _fixed(bp), _fixed(stats)],
        names=["pviHP", "pviLP", "bp", "stats"],
    )
    path = root / split / "part.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(table, path, row_group_size=row_group_size)


def test_population_cache_shapes_and_frozen_partition(tmp_path):
    _write_split(tmp_path, "train", 3)
    _write_split(tmp_path, "test", 2)
    manifest = {
        "schema": "pvi-gcnm-population-6ch-parquet-v1",
        "seed": 42,
        "counts": {"train": 3, "test": 2, "excluded": 0},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "_SUCCESS").write_text("validated\n")

    dataset = PopulationSixChannelParquetDataset(tmp_path, seed=42).build()
    assert dataset.name == "dataset_lazy"
    dataset.set_partition(split_mode="within", random_state=42)
    dataset.set_dataloaders(batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(dataset.get_dataloaders()["train"]))
    assert batch["pviHP"].shape == (2, 2, 40, 40, 250)
    assert batch["pviLP"].shape == (2, 2, 40, 40, 250)
    assert batch["bp"].shape == (2, 50)
    assert batch["stats"].shape == (2, 2, 5)
    state = dataset.state_dict()
    dataset.load_state_dict(state)


def test_pd_targets_retain_original_model_identifiers():
    assert experiment_target("crt", "disjoint").startswith("pd13-crt-")
    assert experiment_target("samba", "disjoint").startswith("pd17-samba-")
    assert experiment_target("crt", "within") == "crt-gcnm6ch-img-to-waveform-exact-pvi"


def test_disjoint_manifest_is_seeded_and_has_no_subject_leakage(tmp_path):
    identities = []
    for subject_index in range(8):
        subject = f"subject{subject_index:03d}"
        for sample_index in range(subject_index + 2):
            identities.append(
                {
                    "sample_id": f"{subject}-{sample_index}",
                    "subject": subject,
                    "source_name": f"source-{subject}",
                    "mask_start": sample_index,
                    "mask_stop": sample_index + 5,
                    "assignment": "train" if sample_index % 2 else "test",
                }
            )
        identities.append(
            {
                "sample_id": f"{subject}-excluded",
                "subject": subject,
                "source_name": f"source-{subject}",
                "mask_start": 100,
                "mask_stop": 105,
                "assignment": "excluded",
            }
        )
    source = tmp_path / "within.json"
    source.write_text(
        json.dumps(
            {
                "schema": "pvi-gcnm-population-within-split-v1",
                "row_count": len(identities),
                "source_count": 8,
                "identities": identities,
            }
        )
    )
    first_path = tmp_path / "disjoint-a.json"
    second_path = tmp_path / "disjoint-b.json"
    first = build_disjoint_manifest(
        source_manifest_path=source,
        output_path=first_path,
        seed=42,
    )
    second = build_disjoint_manifest(
        source_manifest_path=source,
        output_path=second_path,
        seed=42,
    )
    assert first["test_subjects"] == second["test_subjects"]
    assert first["counts"] == second["counts"]
    assert first["counts"]["excluded"] == 8
    for subject, counts in first["subjects"].items():
        assert not (counts["train"] and counts["test"]), subject


def test_disjoint_manifest_can_restore_legacy_all_identity_holdout(tmp_path):
    identities = []
    for subject_index in range(5):
        subject = f"subject{subject_index:03d}"
        identities.extend(
            [
                {
                    "sample_id": f"{subject}-active",
                    "subject": subject,
                    "source_name": subject,
                    "mask_start": 0,
                    "mask_stop": 5,
                    "assignment": "train",
                },
                {
                    "sample_id": f"{subject}-formerly-excluded",
                    "subject": subject,
                    "source_name": subject,
                    "mask_start": 1,
                    "mask_stop": 6,
                    "assignment": "excluded",
                },
            ]
        )
    source = tmp_path / "within.json"
    source.write_text(
        json.dumps(
            {
                "schema": "pvi-gcnm-population-within-split-v1",
                "row_count": len(identities),
                "source_count": 5,
                "identities": identities,
            }
        )
    )
    manifest = build_disjoint_manifest(
        source_manifest_path=source,
        output_path=tmp_path / "legacy.json",
        seed=42,
        requested_subjects=["subject003"],
        all_identities_active=True,
    )
    assert manifest["counts"] == {"train": 8, "test": 2, "excluded": 0}
    assert manifest["active_policy"] == "all-frozen-identities"
    assert manifest["test_subjects"] == ["subject003"]


def test_disjoint_v4_view_reuses_payload_and_replays_schedules(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    additional_root = tmp_path / "completion"
    additional_root.mkdir()
    rows = [
        {
            "sample_id": f"id-{index}",
            "subject": f"subject{index // 2:03d}",
            "session": "baseline",
            "source_name": f"source-{index // 2}",
            "mask_start": index,
            "mask_stop": index + 5,
        }
        for index in range(6)
    ]
    frozen_active = tmp_path / "within.json"
    frozen_active.write_text('{"schema":"pvi-gcnm-population-within-split-v1"}\n')
    frozen_digest = hashlib.sha256(frozen_active.read_bytes()).hexdigest()
    for root, indices in ((source_root, [0, 2, 4]), (additional_root, [1, 3, 5])):
        path = root / "train" / "part.parquet"
        path.parent.mkdir()
        pq.write_table(
            pa.Table.from_pylist([rows[index] for index in indices]),
            path,
            row_group_size=1,
        )
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": (
                        "pvi-gcnm-population-6ch-parquet-v3"
                        if root == source_root
                        else "pvi-gcnm-population-6ch-parquet-v1"
                    ),
                    "seed": 42,
                    "counts": {"train": 3, "test": 0, "excluded": 3},
                    "files": {
                        "train": [str(path.relative_to(root))],
                        "test": [],
                    },
                    "split_manifest_sha256": frozen_digest,
                }
            )
        )
        (root / "_SUCCESS").write_text("validated\n")

    identities = []
    for index, row in enumerate(rows):
        identities.append(
            {
                **row,
                "assignment": "test" if index < 2 else "train",
            }
        )
    split_path = tmp_path / "disjoint.json"
    split_path.write_text(
        json.dumps(
            {
                "schema": "pvi-gcnm-population-disjoint-split-v1",
                "seed": 42,
                "row_count": len(identities),
                "counts": {"train": 4, "test": 2, "excluded": 0},
                "source_split_manifest_sha256": frozen_digest,
                "test_subjects": ["subject000"],
                "train_subjects": ["subject001", "subject002"],
                "identities": identities,
            }
        )
    )
    output = tmp_path / "view"
    manifest = build_disjoint_view(
        source_root=source_root,
        additional_source_roots=[additional_root],
        split_manifest_path=split_path,
        output_root=output,
        epochs=2,
        batch_size=2,
        cluster_size=2,
        seed=42,
        resume=False,
        enforce_scratch=False,
    )
    assert manifest["data_view"]["payload_copied"] is False
    assert len(manifest["data_view"]["source_caches"]) == 2
    assert not list(output.rglob("*.parquet"))
    dataset = PopulationSixChannelParquetDataset(
        output,
        seed=42,
        require_exact_pvi_schedules=True,
    ).build()
    dataset.set_partition(split_mode="disjoint", random_state=42)
    dataset.set_dataloaders(batch_size=2, shuffle=True, num_workers=0)
    assert dataset.index_view
    assert len(dataset.subsets["train"]) == 4
    assert len(dataset.subsets["test"]) == 2
    assert [row["sample_id"] for row in dataset.subsets["test"].metadata_rows(["sample_id"])] == [
        "id-0",
        "id-1",
    ]
    train_selection = set(np.load(output / manifest["data_view"]["selection_files"]["train"]).tolist())
    test_selection = set(np.load(output / manifest["data_view"]["selection_files"]["test"]).tolist())
    assert not train_selection & test_selection
    assert train_selection | test_selection == set(range(6))


def test_completion_builder_projects_skip_cache_sample_ids(tmp_path):
    cache = tmp_path / "cache"
    path = cache / "train" / "part.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"sample_id": "id-a"}, {"sample_id": "id-b"}]),
        path,
    )
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "pvi-gcnm-population-6ch-parquet-v3",
                "files": {"train": ["train/part.parquet"], "test": []},
            }
        )
    )
    (cache / "_SUCCESS").write_text("validated\n")
    assert sample_ids_from_cache(cache) == {"id-a", "id-b"}


def test_row_group_sampler_is_seeded_by_epoch():
    left = RowGroupBatchSampler([(0, 2), (2, 4), (4, 6)], 2, shuffle=True, seed=42)
    right = RowGroupBatchSampler([(0, 2), (2, 4), (4, 6)], 2, shuffle=True, seed=42)
    assert list(left) == list(right)
    assert list(left) == list(right)


def test_v2_cache_requires_its_materialized_batch_size(tmp_path):
    _write_split(tmp_path, "train", 4)
    _write_split(tmp_path, "test", 2)
    manifest = {
        "schema": "pvi-gcnm-population-6ch-parquet-v2",
        "seed": 42,
        "counts": {"train": 4, "test": 2, "excluded": 0},
        "batch_layout": {
            "strategy": "pvi-source-clustered-prestratified",
            "batch_size": 2,
            "cluster_size": 30,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "_SUCCESS").write_text("validated\n")
    dataset = PopulationSixChannelParquetDataset(
        tmp_path, seed=42, require_prestratified=True
    ).build()
    dataset.set_partition(split_mode="within", random_state=42)
    dataset.set_dataloaders(batch_size=2, shuffle=True, num_workers=0)
    assert dataset.prestratified
    with pytest.raises(ValueError, match="fixes batch size"):
        dataset.set_dataloaders(batch_size=4, shuffle=True, num_workers=0)


def test_exact_pvi_order_matches_reference_implementation():
    indices = list(range(18))
    grouping = [list(range(0, 4)), list(range(4, 11)), [], list(range(11, 18))]

    def reference(rng):
        file_subgroups = []
        for group in grouping:
            file_subgroups.append(list(set(indices) & set(group)))
        rng.shuffle(file_subgroups)
        flattened = []
        for first in range(0, len(file_subgroups), 2):
            cluster = file_subgroups[first : first + 2]
            combined = reduce(concat, cluster)
            rng.shuffle(combined)
            flattened.extend(combined)
        return flattened

    assert pvi_stratified_order(
        indices, grouping, cluster_size=2, rng=random.Random(17)
    ) == reference(random.Random(17))


def test_v3_loader_replays_and_resumes_precomputed_epoch_schedules(tmp_path):
    _write_split(tmp_path, "train", 4, row_group_size=1)
    _write_split(tmp_path, "test", 2, row_group_size=1)
    schedule_root = tmp_path / "schedules"
    schedule_root.mkdir()
    np.save(schedule_root / "train_order.npy", np.asarray([[3, 0, 2, 1], [1, 2, 0, 3]], dtype=np.int32))
    np.save(schedule_root / "test_order.npy", np.asarray([[1, 0], [0, 1]], dtype=np.int32))
    manifest = {
        "schema": "pvi-gcnm-population-6ch-parquet-v3",
        "seed": 42,
        "counts": {"train": 4, "test": 2, "excluded": 0},
        "files": {"train": ["train/part.parquet"], "test": ["test/part.parquet"]},
        "row_group_size": 1,
        "batch_layout": {
            "strategy": "precomputed-exact-pvi-batch-schedules",
            "batch_size": 2,
            "cluster_size": 30,
            "epochs": 2,
            "schedule_files": {
                "train": "schedules/train_order.npy",
                "test": "schedules/test_order.npy",
            },
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "_SUCCESS").write_text("validated\n")
    dataset = PopulationSixChannelParquetDataset(
        tmp_path, seed=42, require_exact_pvi_schedules=True
    ).build()
    dataset.set_partition(split_mode="within", random_state=42)
    dataset.set_dataloaders(batch_size=2, shuffle=True, num_workers=0)
    loaders = dataset.get_dataloaders()
    assert isinstance(loaders["train"].batch_sampler, PrecomputedPviBatchSampler)
    assert list(loaders["train"].batch_sampler) == [[3, 0], [2, 1]]
    state = dataset.state_dict()
    assert state["sampler_epochs"] == {"train": 1, "test": 0}

    resumed = PopulationSixChannelParquetDataset(
        tmp_path, seed=42, require_exact_pvi_schedules=True
    ).build()
    resumed.load_state_dict(state)
    resumed.set_partition(split_mode="within", random_state=42)
    resumed.set_dataloaders(batch_size=2, shuffle=True, num_workers=0)
    resumed_loaders = resumed.get_dataloaders()
    assert list(resumed_loaders["train"].batch_sampler) == [[1, 2], [0, 3]]


def test_exact_sampler_does_not_skip_first_schedule_when_iterator_is_discarded(tmp_path):
    schedule_path = tmp_path / "order.npy"
    np.save(schedule_path, np.asarray([[3, 0, 2, 1]], dtype=np.int32))
    sampler = PrecomputedPviBatchSampler(
        schedule_path,
        row_count=4,
        batch_size=2,
    )
    # PyTorch multiprocessing creates one iterator and discards it before
    # consuming a second iterator during initial worker startup.
    _discarded = iter(sampler)
    assert sampler.epoch == 0
    observed = [value for batch in iter(sampler) for value in batch]
    assert observed == [3, 0, 2, 1]
    assert sampler.epoch == 1


def _write_small_repack_split(root, split, source_names):
    rows = []
    payload = 0
    for source_name in source_names:
        for local in range(2):
            rows.append(
                {
                    "payload": payload,
                    "sample_id": f"{split}-{source_name}-{local}",
                    "subject": f"subject-{source_name}",
                    "session": "baseline",
                    "source_name": source_name,
                }
            )
            payload += 1
    table = pa.Table.from_pylist(rows)
    path = root / split / "source.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(table, path, row_group_size=3)
    return path, rows


def test_repack_materializes_mixed_batches_without_changing_rows(tmp_path):
    source = tmp_path / "source"
    train_path, train_rows = _write_small_repack_split(
        source, "train", [f"train{i}" for i in range(8)]
    )
    test_path, test_rows = _write_small_repack_split(
        source, "test", [f"test{i}" for i in range(4)]
    )
    source_manifest = {
        "schema": "pvi-gcnm-population-6ch-parquet-v1",
        "seed": 42,
        "counts": {"train": len(train_rows), "test": len(test_rows), "excluded": 3},
        "files": {
            "train": [str(train_path.relative_to(source))],
            "test": [str(test_path.relative_to(source))],
        },
    }
    (source / "manifest.json").write_text(json.dumps(source_manifest))
    (source / "_SUCCESS").write_text("validated\n")
    output = tmp_path / "stratified"
    manifest = repack_cache(
        source_root=source,
        output_root=output,
        batch_size=4,
        cluster_size=8,
        row_groups_per_file=2,
        workers=1,
        seed=42,
        resume=False,
    )
    assert manifest["schema"] == "pvi-gcnm-population-6ch-parquet-v2"
    assert manifest["counts"] == source_manifest["counts"]
    assert manifest["layout_audit"]["train"]["unique_sources_per_full_batch"]["min"] >= 2

    expected = {
        row["sample_id"]: (row["payload"], row["source_name"])
        for row in train_rows
    }
    observed = {}
    for relative_path in manifest["files"]["train"]:
        parquet = pq.ParquetFile(output / relative_path)
        for group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(group)
            sources = table["source_name"].to_pylist()
            assert len(set(sources)) >= 2
            for row in table.to_pylist():
                observed[row["sample_id"]] = (row["payload"], row["source_name"])
    assert observed == expected
    assert (output / "_SUCCESS").is_file()
    assert not (output / "_INCOMPLETE").exists()


def test_exact_cache_stores_one_copy_and_precomputes_all_epochs(tmp_path):
    source = tmp_path / "source"
    train_path, train_rows = _write_small_repack_split(
        source, "train", [f"train{i}" for i in range(4)]
    )
    test_path, test_rows = _write_small_repack_split(
        source, "test", [f"test{i}" for i in range(2)]
    )
    identities = []
    for index in range(max(len(train_rows), len(test_rows))):
        if index < len(train_rows):
            identities.append({**train_rows[index], "assignment": "train"})
        if index < len(test_rows):
            identities.append({**test_rows[index], "assignment": "test"})
    identities.append(
        {
            "sample_id": "excluded-sample",
            "source_name": "excluded-source",
            "assignment": "excluded",
        }
    )
    split_manifest_path = tmp_path / "frozen_split.json"
    split_manifest_path.write_text(
        json.dumps(
            {
                "schema": "pvi-gcnm-population-within-split-v1",
                "row_count": len(identities),
                "identities": identities,
            }
        )
    )
    split_digest = hashlib.sha256(split_manifest_path.read_bytes()).hexdigest()
    source_manifest = {
        "schema": "pvi-gcnm-population-6ch-parquet-v1",
        "seed": 42,
        "counts": {"train": len(train_rows), "test": len(test_rows), "excluded": 1},
        "files": {
            "train": [str(train_path.relative_to(source))],
            "test": [str(test_path.relative_to(source))],
        },
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": split_digest,
    }
    (source / "manifest.json").write_text(json.dumps(source_manifest))
    (source / "_SUCCESS").write_text("validated\n")
    output = tmp_path / "exact"
    manifest = build_exact_cache(
        source_root=source,
        output_root=output,
        workers=1,
        epochs=3,
        batch_size=4,
        cluster_size=2,
        seed=42,
        resume=False,
        enforce_scratch=False,
    )
    assert manifest["schema"] == "pvi-gcnm-population-6ch-parquet-v3"
    assert manifest["counts"] == source_manifest["counts"]
    assert manifest["row_group_size"] == 1
    assert (
        manifest["batch_layout"]["reference_index_layout"]["index_space"]
        == "original-pvi-global-indices"
    )
    source_groups = {}
    for global_index, identity in enumerate(identities):
        source_groups.setdefault(identity["source_name"], []).append(global_index)
    reference_rng = random.Random(42)
    for split in ("train", "test"):
        total_rows = 0
        total_groups = 0
        for relative_path in manifest["files"][split]:
            metadata = pq.ParquetFile(output / relative_path).metadata
            total_rows += metadata.num_rows
            total_groups += metadata.num_row_groups
        assert total_groups == total_rows == source_manifest["counts"][split]
        schedules = np.load(output / manifest["batch_layout"]["schedule_files"][split])
        assert schedules.shape == (3, source_manifest["counts"][split])
        selected_global = [
            index
            for index, identity in enumerate(identities)
            if identity["assignment"] == split
        ]
        global_order = pvi_stratified_order(
            selected_global,
            list(source_groups.values()),
            cluster_size=2,
            rng=reference_rng,
        )
        local_by_global = {
            global_index: local_index
            for local_index, global_index in enumerate(selected_global)
        }
        assert schedules[0].tolist() == [
            local_by_global[global_index] for global_index in global_order
        ]
        for epoch_order in schedules:
            assert sorted(epoch_order.tolist()) == list(range(source_manifest["counts"][split]))


def test_production_cache_builder_rejects_non_scratch_output(tmp_path):
    with pytest.raises(ValueError, match="must remain under"):
        require_scratch_path(tmp_path)


def test_cache_worker_pairs_newton_and_gcnm_in_contract_order(tmp_path):
    image_width = 40 * 40 * 250
    s1 = np.full((1, image_width), 2.0, dtype=np.float32)
    s2 = np.full((1, image_width), 3.0, dtype=np.float32)
    coordinate = pa.Table.from_arrays(
        [
            _fixed(s1),
            _fixed(s2),
            _fixed(np.arange(50, dtype=np.float32)[None]),
            _fixed(np.arange(10, dtype=np.float32)[None]),
            pa.array(["id"]), pa.array(["subject001"]), pa.array(["baseline"]),
            pa.array(["source"]), pa.array([0], type=pa.int32()),
            pa.array([5], type=pa.int32()),
        ],
        names=[
            "s1", "s2", "bp_waveform", "stats", "sample_id", "subject",
            "session", "source_name", "mask_start", "mask_stop",
        ],
    )
    coordinate_path = tmp_path / "coordinate.parquet"
    pq.write_table(coordinate, coordinate_path)
    source_path = tmp_path / "source.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("data/pviHP/img", data=np.full((40, 40, 250), 4.0, dtype=np.float32))
        handle.create_dataset("data/pviLP/img", data=np.full((40, 40, 250), 5.0, dtype=np.float32))
    output = tmp_path / "cache"
    report = process_shard(
        {
            "coordinate_path": str(coordinate_path),
            "output_root": str(output),
            "assignments": {"id": "train"},
            "sources": {"source": str(source_path)},
            "row_group_size": 64,
        }
    )
    assert report["counts"] == {"train": 1, "test": 0, "excluded": 0}
    table = pq.read_table(report["outputs"]["train"])
    hp = np.asarray(table["pviHP"].combine_chunks().values).reshape(1, 2, 40, 40, 250)
    lp = np.asarray(table["pviLP"].combine_chunks().values).reshape(1, 2, 40, 40, 250)
    assert np.all(hp[:, 0] == 4.0)
    assert np.all(hp[:, 1] == 2.0)
    assert np.all(lp[:, 0] == 5.0)
    assert np.all(lp[:, 1] == 3.0)


def test_six_channel_preprocessor_contract():
    class FakeModel:
        nan_values = 0.0

        @staticmethod
        def _compute_diff(value):
            return value + 10

    hp = torch.stack((torch.full((2, 3), 1.0), torch.full((2, 3), 2.0)), dim=0)[None]
    lp = torch.stack((torch.full((2, 3), 3.0), torch.full((2, 3), 4.0)), dim=0)[None]
    result = six_channel_process_sequence(FakeModel(), {"pviHP": hp, "pviLP": lp})
    expected = [1.0, 13.0, 23.0, 2.0, 4.0, 14.0]
    assert result.shape[1] == 6
    assert [float(result[0, index, 0, 0]) for index in range(6)] == expected
