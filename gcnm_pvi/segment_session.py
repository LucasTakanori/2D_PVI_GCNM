"""Segment production PVI trials into cardiac periods matching getEnsemble.m."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.signal import find_peaks

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.production_preprocess import process_vmeas_production
from gcnm_pvi.runtime import load_meshes


def matlab_ensemble_vmeas(mat_path: Path | str, elec_configs) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load raw voltages cached by SCIOSPEC and apply its remap/make_eit rules."""
    ensemble = loadmat(
        Path(mat_path), simplify_cells=True, variable_names=["ensemble_out"]
    )["ensemble_out"]
    frames = ensemble["frames"]
    if isinstance(frames, dict):
        frames = [frames]
    voltage = np.stack([np.asarray(frame["meas"]["voltage"]) for frame in frames])

    # get_channel_parity sees stimulation electrodes 1..8, so channel_select
    # returns those same 1-based channel IDs. In Python these are columns 0:8.
    voltage = voltage[:, :, : elec_configs.num_stim]
    M = elec_configs.potential_config.extractor
    blocks = []
    for stim in range(elec_configs.num_stim):
        blocks.append(M[stim] @ voltage[:, stim, :].T)
    vmeas = np.concatenate(blocks, axis=0)

    tvec = np.asarray(ensemble["time"]["vec"], dtype=np.float64).reshape(-1)
    fs = float(ensemble["time"]["sampling_rate"])
    raw_current = float(ensemble["stim"]["current"])
    return vmeas, tvec, fs, raw_current


def detect_period_boundaries(
    signal: np.ndarray,
    fs: float,
    *,
    min_peak_distance_s: float = 0.5,
    prominence_factor: float | None = 2.0,
) -> np.ndarray:
    """Detect minima using MATLAB findpeaks-compatible distance/prominence."""
    y = np.asarray(signal, dtype=np.float64).reshape(-1)
    inverted = -y
    mad1 = float(np.median(np.abs(inverted - np.median(inverted))))
    prominence = None if prominence_factor is None else prominence_factor * mad1
    peaks, _ = find_peaks(
        inverted,
        distance=max(1, int(np.floor(min_peak_distance_s * fs))),
        prominence=prominence,
    )
    if peaks.size < 2:
        raise ValueError("Fewer than two PVI peaks detected")
    return peaks


def interpolate_periods(data: np.ndarray, boundaries: np.ndarray, points: int = 50) -> np.ndarray:
    """Crop peak-to-peak periods and linearly resample each to ``points``."""
    arr = np.asarray(data)
    rows = arr.reshape(-1, arr.shape[-1])
    q = np.linspace(0.0, 1.0, points)
    periods = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        segment = rows[:, start : stop + 1]
        x = np.linspace(0.0, 1.0, segment.shape[1])
        periods.append(interp1d(x, segment, axis=1, kind="linear")(q))
    stacked = np.stack(periods, axis=1)  # rows, periods, points
    return stacked.reshape(*arr.shape[:-1], stacked.shape[1], points)


def segment_trial(
    mat_path: Path,
    cfg: GcnmConfig,
    elec_configs,
    points: int = 50,
    *,
    min_peak_distance_s: float = 0.5,
    prominence_factor: float | None = 2.0,
) -> dict:
    vmeas, tvec, fs, raw_current = matlab_ensemble_vmeas(mat_path, elec_configs)
    if not np.isclose(abs(cfg.stim_current), abs(raw_current), rtol=1e-6, atol=1e-12):
        raise ValueError(
            f"{mat_path}: config current {cfg.stim_current} A != raw {raw_current} A"
        )
    proc = process_vmeas_production(
        vmeas,
        fs=round(fs),
        stim_current=cfg.stim_current,
        filter_cutoff_hz=cfg.filter_cutoff_hz,
        hp_lp_window=cfg.hp_lp_window,
    )
    hp = proc["pviHP"]
    lp = proc["pviLP"]
    signal = np.mean(hp["resistance"], axis=0)
    boundaries = detect_period_boundaries(
        signal,
        fs,
        min_peak_distance_s=min_peak_distance_s,
        prominence_factor=prominence_factor,
    )

    trim = vmeas.shape[1] - hp["resistance"].shape[1]
    tvec_trim = tvec[trim:]
    duration = np.diff(tvec_trim[boundaries])
    return {
        "name": mat_path.parent.name,
        "mat_path": str(mat_path),
        "fs": fs,
        "trim": trim,
        "boundaries": boundaries,
        "boundary_times": tvec_trim[boundaries],
        "duration": duration,
        "pviHP_resistance": interpolate_periods(hp["resistance"], boundaries, points),
        "pviHP_reactance": interpolate_periods(hp["reactance"], boundaries, points),
        "pviLP_resistance": interpolate_periods(lp["resistance"], boundaries, points),
        "pviLP_reactance": interpolate_periods(lp["reactance"], boundaries, points),
    }


def discover_trial_mats(session_dir: Path | str) -> list[Path]:
    session_dir = Path(session_dir)
    return sorted(session_dir.glob("*/bioz_*_full.mat"), key=lambda p: p.parent.name)


def segment_session(
    session_dir: Path | str,
    cfg: GcnmConfig,
    *,
    points: int = 50,
    selected_trials: set[int] | None = None,
    min_peak_distance_s: float = 0.5,
    prominence_factor: float | None = 2.0,
) -> dict:
    _, _, elec_configs, _, _ = load_meshes(cfg)
    mats = discover_trial_mats(session_dir)
    if not mats:
        raise FileNotFoundError(f"No bioz_*_full.mat trials under {session_dir}")

    trials = []
    for trial_index, mat_path in enumerate(mats, start=1):
        if selected_trials is not None and trial_index not in selected_trials:
            continue
        trial = segment_trial(
            mat_path,
            cfg,
            elec_configs,
            points=points,
            min_peak_distance_s=min_peak_distance_s,
            prominence_factor=prominence_factor,
        )
        trial["trial_index"] = trial_index
        trials.append(trial)
        print(f"trial {trial_index:02d} {trial['name']}: {len(trial['duration'])} periods")

    out: dict[str, object] = {
        "trials": trials,
        "points": points,
        "min_peak_distance_s": min_peak_distance_s,
        "prominence_factor": prominence_factor,
    }
    for component in ("pviHP", "pviLP"):
        for field in ("resistance", "reactance"):
            key = f"{component}_{field}"
            blocks = [np.asarray(t[key]) for t in trials]
            # Each block is channels, periods, points. Flatten in period order.
            out[key] = np.concatenate(blocks, axis=1)
    out["duration"] = np.concatenate([np.asarray(t["duration"]) for t in trials])
    out["trial_index"] = np.concatenate(
        [np.full(len(t["duration"]), t["trial_index"], dtype=np.int16) for t in trials]
    )
    return out


def save_segmented(result: dict, out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value
        for key, value in result.items()
        if isinstance(value, np.ndarray)
    }
    np.savez_compressed(out_dir / "segmented_periods.npz", **arrays)
    metadata = {
        "points": result["points"],
        "min_peak_distance_s": result["min_peak_distance_s"],
        "prominence_factor": result["prominence_factor"],
        "num_periods": int(len(result["duration"])),
        "trials": [
            {
                "trial_index": t["trial_index"],
                "name": t["name"],
                "mat_path": t["mat_path"],
                "fs": t["fs"],
                "trim": t["trim"],
                "num_periods": int(len(t["duration"])),
            }
            for t in result["trials"]
        ],
    }
    (out_dir / "segmentation_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return out_dir / "segmented_periods.npz"


def compare_segmented_to_hdf5(result: dict, h5_path: Path | str) -> dict:
    report: dict[str, object] = {}
    with h5py.File(h5_path, "r") as f:
        ref_periods = int(f["metadata/num_periods"][()].item())
        report["pred_periods"] = int(len(result["duration"]))
        report["h5_periods"] = ref_periods
        report["period_count_match"] = report["pred_periods"] == ref_periods
        for component in ("pviHP", "pviLP"):
            for field in ("resistance", "reactance"):
                key = f"{component}_{field}"
                pred = np.asarray(result[key]).reshape(32, -1)
                ref = np.asarray(f[f"data/{component}/{field}"])
                n = min(pred.shape[1], ref.shape[1])
                a, b = pred[:, :n], ref[:, :n]
                corr = [
                    float(np.corrcoef(a[ch], b[ch])[0, 1])
                    for ch in range(min(a.shape[0], b.shape[0]))
                ]
                report[key] = {
                    "compared_values_per_channel": n,
                    "mse": float(np.mean((a - b) ** 2)),
                    "mean_channel_corr": float(np.mean(corr)),
                    "per_channel_corr": corr,
                }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Segment PVI trials into 50-point cardiac periods")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--h5", type=Path, default=None)
    p.add_argument("--points", type=int, default=50)
    p.add_argument("--min-peak-distance", type=float, default=0.5)
    p.add_argument(
        "--prominence-factor",
        type=float,
        default=2.0,
        help="Multiplier on median MAD; negative disables prominence filtering",
    )
    p.add_argument("--trials", default="", help="Comma-separated 1-based trial indices")
    args = p.parse_args()
    selected = {int(x) for x in args.trials.split(",") if x.strip()} or None
    cfg = GcnmConfig.from_yaml(args.config)
    prominence = None if args.prominence_factor < 0 else args.prominence_factor
    result = segment_session(
        args.session_dir,
        cfg,
        points=args.points,
        selected_trials=selected,
        min_peak_distance_s=args.min_peak_distance,
        prominence_factor=prominence,
    )
    path = save_segmented(result, args.out_dir)
    print(f"Wrote {path}")
    if args.h5:
        report = compare_segmented_to_hdf5(result, args.h5)
        text = json.dumps(report, indent=2)
        print(text)
        (args.out_dir / "segmentation_validation.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
