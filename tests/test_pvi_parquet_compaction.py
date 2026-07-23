import json

import pyarrow as pa
import pyarrow.parquet as pq

from gcnm_pvi.export_pvi_parquet import compact_session_shards


def test_compaction_preserves_order_bounds_rows_and_removes_staging(tmp_path):
    output_root = tmp_path / "coordinate_v2"
    staging_root = output_root / "session_shards"
    staging_root.mkdir(parents=True)
    schema = pa.schema(
        [pa.field("source_name", pa.string()), pa.field("sequence", pa.int32())]
    )
    reports = []
    expected = []
    for source_name, count in (("session_a", 3), ("session_b", 4), ("session_c", 2)):
        path = staging_root / f"{source_name}.parquet"
        values = list(range(len(expected), len(expected) + count))
        table = pa.Table.from_arrays(
            [pa.array([source_name] * count), pa.array(values, type=pa.int32())],
            schema=schema,
        )
        pq.write_table(table, path, compression="zstd", row_group_size=2)
        reports.append(
            {
                "source_name": source_name,
                "rows": count,
                "staging_shard": str(path.resolve()),
            }
        )
        expected.extend(values)

    compacted, inventory = compact_session_shards(
        reports, output_root, target_rows=4, batch_rows=2
    )

    assert [item["rows"] for item in inventory] == [4, 4, 1]
    final_paths = [item["path"] for item in inventory]
    tables = [pq.read_table(path) for path in final_paths]
    merged = pa.concat_tables(tables)
    assert merged["sequence"].to_pylist() == expected
    assert not staging_root.exists()
    assert all("staging_shard" not in report for report in compacted)
    assert "session_shards" not in json.dumps(compacted)
    assert [len(report["shards"]) for report in compacted] == [1, 2, 2]
    assert all(path.startswith(str((output_root / "shards").resolve())) for path in final_paths)
