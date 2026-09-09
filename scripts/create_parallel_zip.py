#!/usr/bin/env python3
"""Create and CRC-verify a standard ZIP64 archive with parallel file reads."""

from __future__ import annotations

import argparse
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


def _files(source: Path) -> list[Path]:
    return sorted(path for path in source.rglob("*") if path.is_file())


def _read_entry(path: Path, parent: Path) -> tuple[zipfile.ZipInfo, bytes]:
    archive_name = path.relative_to(parent).as_posix()
    info = zipfile.ZipInfo.from_file(path, arcname=archive_name)
    info.compress_type = zipfile.ZIP_STORED
    return info, path.read_bytes()


def create(source: Path, output: Path, workers: int, in_flight: int) -> None:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir() or not (source / "_SUCCESS").is_file():
        raise FileNotFoundError(f"validated source directory is missing: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    paths = _files(source)
    if not paths:
        raise ValueError(f"source contains no files: {source}")
    parent = source.parent
    started = time.time()
    written_bytes = 0
    completed = 0
    iterator = iter(paths)
    pending = set()
    with ThreadPoolExecutor(max_workers=workers) as pool, zipfile.ZipFile(
        output,
        mode="x",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for _ in range(min(in_flight, len(paths))):
            path = next(iterator)
            pending.add(pool.submit(_read_entry, path, parent))
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                info, data = future.result()
                archive.writestr(info, data, compress_type=zipfile.ZIP_STORED)
                written_bytes += len(data)
                completed += 1
                try:
                    path = next(iterator)
                except StopIteration:
                    pass
                else:
                    pending.add(pool.submit(_read_entry, path, parent))
                if completed % 1000 == 0 or completed == len(paths):
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        f"archive_progress files={completed}/{len(paths)} "
                        f"input_gib={written_bytes / 2**30:.3f} "
                        f"rate_mib_s={written_bytes / 2**20 / elapsed:.2f}",
                        flush=True,
                    )

    print("archive_write_complete", flush=True)
    with zipfile.ZipFile(output, mode="r", allowZip64=True) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"CRC verification failed at {bad}")
        members = len(archive.infolist())
    if members != len(paths):
        raise RuntimeError(f"archive contains {members} files, expected {len(paths)}")
    print(f"archive_crc_pass files={members}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--in-flight", type=int, default=64)
    args = parser.parse_args()
    if args.workers < 1 or args.in_flight < args.workers:
        parser.error("workers must be positive and in-flight must be at least workers")
    create(args.source, args.output, args.workers, args.in_flight)


if __name__ == "__main__":
    main()
