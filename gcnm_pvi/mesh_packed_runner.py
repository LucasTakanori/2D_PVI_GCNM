"""Run mesh data-generation or GCNM-training tasks in a bounded local pool."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RINGS = (
    "US060", "US065", "US070", "US075", "US080", "US085", "US090", "US095",
    "US100", "US105", "US110", "US115", "US120", "US125", "US130",
)


def _data_required(task: dict) -> list[Path]:
    dataset = Path(task["dataset"])
    names = [
        "train.npz", "validation.npz", "test.npz", "metadata.json",
        "train_anatomy.json", "validation_anatomy.json", "test_anatomy.json",
    ]
    if task["family"] == "diffusion":
        names.append("default_finger_prior.npz")
    return [dataset / name for name in names]


def _training_required(task: dict) -> list[Path]:
    models = Path(task["models"])
    model_name = task["model_name"]
    suffixes = ("0.pt", "1.pt") if task["family"] == "coordinate" else (
        "localizer.pt", "refiner.pt"
    )
    result = Path(
        task.get(
            "results",
            ROOT / "data/mesh_training_results" / task["family"] / task["ring"],
        )
    )
    return [
        *(models / f"{model_name}_{suffix}" for suffix in suffixes),
        models / "run_manifest.json",
        result / f"{model_name}_training_report.json",
    ]


def task_state(task: dict, stage: str) -> str:
    """Return missing, complete, or partial without modifying any artifact."""
    required = _data_required(task) if stage == "data" else _training_required(task)
    present = [path.is_file() for path in required]
    if all(present):
        return "complete"
    roots = [Path(task["dataset"])] if stage == "data" else [
        Path(task["models"]),
        Path(
            task.get(
                "results",
                ROOT
                / "data/mesh_training_results"
                / task["family"]
                / task["ring"],
            )
        ),
    ]
    if any(present) or any(path.exists() for path in roots):
        return "partial"
    return "missing"


def _visible_gpus(count: int) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if not devices:
        devices = [str(index) for index in range(count)]
    if len(devices) < count:
        raise RuntimeError(f"only {len(devices)} visible GPUs; requested {count}")
    return devices[:count]


def task_environment(
    task: dict, stage: str, task_cpus: int, gpu: str | None
) -> dict[str, str]:
    """Build the explicit identity and resource contract for one task."""

    environment = os.environ.copy()
    environment.update(
        {
            "REPO_ROOT": str(ROOT),
            "FAMILY": task["family"],
            "RING": task["ring"],
            "SLURM_ARRAY_TASK_ID": str(RINGS.index(task["ring"])),
            "TASK_CPUS": str(task_cpus),
            "OMP_NUM_THREADS": str(task_cpus),
            "MKL_NUM_THREADS": str(task_cpus),
            "OPENBLAS_NUM_THREADS": str(task_cpus),
            "NUMEXPR_NUM_THREADS": str(task_cpus),
            "GCNM_PHYSICS_WORKERS": str(task_cpus),
        }
    )
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu
    return environment


def _run_task(task: dict, stage: str, task_cpus: int, gpu: str | None) -> None:
    if stage == "data":
        script = ROOT / "slurm/launch_generate_mesh_representation_data.sh"
    elif "beats1000x50_v1" in str(task["dataset"]):
        script = ROOT / "slurm/launch_train_mesh_beats50_model.sh"
    else:
        script = ROOT / "slurm/launch_train_mesh_representation.sh"
    log_dir = ROOT / "logs/mesh_packed" / stage
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task['family']}_{task['ring']}.out"
    environment = task_environment(task, stage, task_cpus, gpu)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            ["bash", str(script)],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"{stage} failed for {task['family']}/{task['ring']} "
            f"with exit {result.returncode}; see {log_path}"
        )


def run_pool(tasks: list[dict], stage: str, workers: int, task_cpus: int, gpus: int) -> None:
    pending: queue.Queue[dict] = queue.Queue()
    for task in tasks:
        state = task_state(task, stage)
        if state == "partial":
            raise RuntimeError(
                f"refusing partial immutable {stage} output for "
                f"{task['family']}/{task['ring']}"
            )
        if state == "complete":
            print(f"skip complete {stage}: {task['family']}/{task['ring']}", flush=True)
        else:
            pending.put(task)
    if pending.empty():
        print(f"all {stage} tasks are already complete", flush=True)
        return

    devices = _visible_gpus(gpus) if stage == "train" else []
    if stage == "train" and workers % gpus:
        raise ValueError("training workers must divide evenly across GPUs")
    failures: list[Exception] = []
    lock = threading.Lock()

    def worker(worker_index: int) -> None:
        gpu = devices[worker_index % gpus] if stage == "train" else None
        while True:
            with lock:
                if failures:
                    return
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            label = f" gpu={gpu}" if gpu is not None else ""
            print(
                f"start {stage}: {task['family']}/{task['ring']}{label}",
                flush=True,
            )
            try:
                _run_task(task, stage, task_cpus, gpu)
                print(f"complete {stage}: {task['family']}/{task['ring']}", flush=True)
            except Exception as error:  # preserve every completed immutable task
                with lock:
                    failures.append(error)
                return
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise failures[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=["data", "train"], required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--task-cpus", type=int, required=True)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks = payload["runs"]
    if len(tasks) != 30:
        raise ValueError(f"expected 30 mesh tasks, found {len(tasks)}")
    if args.stage == "data" and args.gpus:
        raise ValueError("data generation is CPU-only and must not request GPUs")
    if args.stage == "train" and args.gpus <= 0:
        raise ValueError("training requires at least one GPU")
    if args.preflight:
        states = {"complete": 0, "missing": 0, "partial": 0}
        for task in tasks:
            states[task_state(task, args.stage)] += 1
        print(json.dumps({"stage": args.stage, **states}, sort_keys=True))
        if states["partial"]:
            raise RuntimeError(f"found {states['partial']} partial immutable outputs")
        return
    run_pool(tasks, args.stage, args.workers, args.task_cpus, args.gpus)


if __name__ == "__main__":
    main()
