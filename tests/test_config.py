from pathlib import Path

from gcnm_pvi.config import GcnmConfig


def test_missing_relative_solver_root_falls_back_to_environment(
    tmp_path: Path, monkeypatch
) -> None:
    environment_root = tmp_path / "environment-solver"
    environment_root.mkdir()
    config_dir = tmp_path / "checkout" / "configs"
    config_dir.mkdir(parents=True)
    monkeypatch.setenv("PVI_SOLVER_ROOT", str(environment_root))

    config = GcnmConfig.from_dict(
        {"pvi_solver_root": "../../missing-solver"}, base_dir=config_dir
    )

    assert config.pvi_solver_root == environment_root


def test_existing_relative_solver_root_remains_authoritative(
    tmp_path: Path, monkeypatch
) -> None:
    environment_root = tmp_path / "environment-solver"
    environment_root.mkdir()
    config_dir = tmp_path / "checkout" / "configs"
    configured_root = config_dir / "relative-solver"
    configured_root.mkdir(parents=True)
    monkeypatch.setenv("PVI_SOLVER_ROOT", str(environment_root))

    config = GcnmConfig.from_dict(
        {"pvi_solver_root": "relative-solver"}, base_dir=config_dir
    )

    assert config.pvi_solver_root == configured_root.resolve()


def test_explicit_absolute_solver_root_remains_authoritative(
    tmp_path: Path, monkeypatch
) -> None:
    environment_root = tmp_path / "environment-solver"
    environment_root.mkdir()
    configured_root = tmp_path / "absolute-solver"
    monkeypatch.setenv("PVI_SOLVER_ROOT", str(environment_root))

    config = GcnmConfig.from_dict(
        {"pvi_solver_root": str(configured_root)}, base_dir=tmp_path
    )

    assert config.pvi_solver_root == configured_root


def test_absent_solver_root_uses_environment(tmp_path: Path, monkeypatch) -> None:
    environment_root = tmp_path / "environment-solver"
    environment_root.mkdir()
    monkeypatch.setenv("PVI_SOLVER_ROOT", str(environment_root))

    config = GcnmConfig.from_dict({}, base_dir=tmp_path)

    assert config.pvi_solver_root == environment_root
