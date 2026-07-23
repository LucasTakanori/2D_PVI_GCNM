#!/usr/bin/env python3
"""Run the 364 independent CRT tasks with four processes per visible GPU."""

from __future__ import annotations

import argparse
import csv
import os
import queue
import subprocess
import threading
from pathlib import Path


def _visible_gpus(count: int) -> list[str]:
    devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if not devices:
        devices = [str(index) for index in range(count)]
    if len(devices) != count:
        raise RuntimeError(f"expected {count} visible GPUs, found {devices}")
    return devices


def _complete(artifact_root: Path, row: dict[str, str]) -> bool:
    main = artifact_root / row["target"] / "main"
    subject = row["subject"]
    required = (
        main / "checkpoints" / f"{subject}_checkpoints.pth",
        main / "configs" / f"{subject}_configs.json",
        main / "history" / f"{subject}_history.csv",
        main / "results" / f"{subject}_results.csv",
        main / "statistics" / f"{subject}_statistics.json",
        main / "gifs" / f"{subject}_verification.json",
    )
    return all(path.is_file() for path in required)


def _postprocess_ready(artifact_root: Path, row: dict[str, str]) -> bool:
    """Training ended and native result export started, but final QA did not."""

    main = artifact_root / row["target"] / "main"
    subject = row["subject"]
    return (
        (main / "checkpoints" / f"{subject}_checkpoints.pth").is_file()
        and (main / "history" / f"{subject}_history.csv").is_file()
        and (main / "results" / f"{subject}_results.csv").is_file()
        and not (main / "statistics" / f"{subject}_statistics.json").is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    if args.workers % args.gpus:
        raise ValueError("workers must divide evenly across GPUs")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 364 or [int(row["task_id"]) for row in rows] != list(range(364)):
        raise ValueError("packed runner requires the frozen contiguous 364-task matrix")
    pending: queue.Queue[dict[str, str]] = queue.Queue()
    for row in rows:
        if _complete(args.artifact_root, row):
            print(f"skip complete task {row['task_id']} {row['subject']} {row['target']}", flush=True)
        else:
            pending.put(row)
    devices = _visible_gpus(args.gpus)
    args.logs.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        gpu = devices[index % args.gpus]
        while True:
            with lock:
                if failures:
                    return
            try:
                row = pending.get_nowait()
            except queue.Empty:
                return
            task = int(row["task_id"])
            stem = args.logs / f"task_{task:03d}_{row['subject']}_{row['channel_mode']}_{row['output_mode']}"
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "SLURM_ARRAY_TASK_ID": str(task),
                    "SLURM_CPUS_PER_TASK": "4",
                    # Main training process plus three persistent Parquet
                    # workers gives four CPU lanes per CRT and 64 lanes total.
                    "BP_NUM_WORKERS": "3",
                    "BP_OMP_THREADS": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
            )
            if _postprocess_ready(args.artifact_root, row):
                environment["BP_POSTPROCESS_ONLY"] = "1"
            print(f"start task={task} gpu={gpu} {row['subject']} {row['target']}", flush=True)
            with stem.with_suffix(".out").open("w", encoding="utf-8") as stdout, stem.with_suffix(".err").open("w", encoding="utf-8") as stderr:
                result = subprocess.run(
                    ["bash", str(args.launcher)],
                    cwd=args.launcher.resolve().parents[1],
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            if result.returncode:
                message = f"task {task} failed with exit {result.returncode}; see {stem}.err"
                with lock:
                    failures.append(message)
                print(message, flush=True)
            else:
                print(f"complete task={task} gpu={gpu}", flush=True)
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(failures[0])


if __name__ == "__main__":
    main()
