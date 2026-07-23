from pathlib import Path

import gcnm_pvi.mesh_packed_runner as runner


def _task(tmp_path: Path, family: str = "coordinate") -> dict:
    return {
        "family": family,
        "ring": "US060",
        "model_name": f"{family}_b045_US060_seed0",
        "dataset": str(tmp_path / "data" / family / "US060"),
        "models": str(tmp_path / "models" / family / "US060"),
    }


def test_data_state_is_missing_then_partial(tmp_path):
    task = _task(tmp_path)
    assert runner.task_state(task, "data") == "missing"
    dataset = Path(task["dataset"])
    dataset.mkdir(parents=True)
    (dataset / "train.npz").touch()
    assert runner.task_state(task, "data") == "partial"


def test_complete_data_state_requires_diffusion_prior(tmp_path):
    task = _task(tmp_path, "diffusion")
    for path in runner._data_required(task):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert runner.task_state(task, "data") == "complete"
    (Path(task["dataset"]) / "default_finger_prior.npz").unlink()
    assert runner.task_state(task, "data") == "partial"


def test_task_environment_propagates_ring_family_gpu_and_cpu_lanes(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    task = _task(Path("/tmp"), "global_voltage_slots")
    task["ring"] = "US085"
    environment = runner.task_environment(task, "train", task_cpus=4, gpu="2")
    assert environment["FAMILY"] == "global_voltage_slots"
    assert environment["RING"] == "US085"
    assert environment["SLURM_ARRAY_TASK_ID"] == "5"
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert environment["GCNM_PHYSICS_WORKERS"] == "4"
