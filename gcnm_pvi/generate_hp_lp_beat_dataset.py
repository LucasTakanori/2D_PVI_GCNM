#!/usr/bin/env python3
"""Generate absolute full-band and derived HP/LP data for one ring mesh.

One virtual anatomy produces seven consecutive 50-sample beats.  HP/LP are
computed on the full 350-frame sequence, then the padding beat at each end is
discarded.  The retained five beats therefore form exactly the same 250-frame
context consumed by ``mask05`` BP models without moving-mean boundary leakage.

The canonical full-band pair is absolute filtered electrode voltage to true
absolute anatomy conductivity, one frame to one reconstruction.  Referenced
HP/LP conductivity fields are retained as auditable derived targets, but the
current pilot reconstructs the absolute sequence first and decomposes that
predicted conductivity afterward.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gcnm_pvi.anatomical_phantoms import domain_transform
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_differential import production_reconstruction_matrix
from gcnm_pvi.generate_finger_simulator_dataset import (
    _fill_outside,
    _load_simulator,
    _sha256,
)
from gcnm_pvi.generate_multisubject_beat_dataset import (
    _rank_one_template,
    _shifted_vessel_record,
)
from gcnm_pvi.hp_lp_signals import (
    PRODUCTION_MOVMEAN_WINDOW,
    butterworth_lowpass_5hz,
    referenced_hp_lp,
)
from gcnm_pvi.runtime import build_runtime


FRAMES_PER_BEAT = 50
RETAINED_BEATS = 5
PADDING_BEATS = 1
TOTAL_BEATS = RETAINED_BEATS + 2 * PADDING_BEATS


@dataclass(frozen=True)
class SplitSpec:
    name: str
    anatomy_ids: np.ndarray
    mode: str
    seed: int


def split_anatomy_ids(
    train: int,
    validation: int,
    test: int,
    *,
    seed: int,
) -> tuple[SplitSpec, SplitSpec, SplitSpec]:
    """Create disjoint virtual-anatomy partitions before frame expansion."""

    counts = (int(train), int(validation), int(test))
    if any(value <= 0 for value in counts):
        raise ValueError("train, validation, and test anatomy counts must be positive")
    ids = np.arange(sum(counts), dtype=np.int32)
    np.random.default_rng(seed).shuffle(ids)
    a, b = counts[0], counts[0] + counts[1]
    return (
        SplitSpec("train", np.sort(ids[:a]), "linearized", seed + 101),
        SplitSpec("validation", np.sort(ids[a:b]), "nonlinear", seed + 202),
        SplitSpec("test", np.sort(ids[b:]), "nonlinear", seed + 303),
    )


def retained_frame_indices(
    *,
    frames_per_beat: int = FRAMES_PER_BEAT,
    retained_beats: int = RETAINED_BEATS,
    padding_beats: int = PADDING_BEATS,
) -> np.ndarray:
    if frames_per_beat <= 0 or retained_beats <= 0 or padding_beats < 0:
        raise ValueError("beat/frame counts are invalid")
    start = padding_beats * frames_per_beat
    return np.arange(start, start + retained_beats * frames_per_beat, dtype=np.int32)


def _frame_beat_templates(voltage: np.ndarray) -> np.ndarray:
    values = np.asarray(voltage, dtype=np.float64)
    if len(values) % FRAMES_PER_BEAT:
        raise ValueError("template input must contain complete 50-sample beats")
    output = np.empty_like(values)
    for start in range(0, len(values), FRAMES_PER_BEAT):
        template = _rank_one_template(values[start : start + FRAMES_PER_BEAT])
        output[start : start + FRAMES_PER_BEAT] = template
    return output


def decompose_and_retain(
    sigma_absolute: np.ndarray,
    voltage_filtered: np.ndarray,
    voltage_clean_filtered: np.ndarray,
    vessel_amplitudes: np.ndarray,
    *,
    movmean_window: int = PRODUCTION_MOVMEAN_WINDOW,
    sigma_resting: np.ndarray | None = None,
    voltage_raw: np.ndarray | None = None,
    voltage_clean_raw: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Retain true absolute/full fields and the referenced HP/LP components."""

    expected = TOTAL_BEATS * FRAMES_PER_BEAT
    arrays = (
        np.asarray(sigma_absolute),
        np.asarray(voltage_filtered),
        np.asarray(voltage_clean_filtered),
        np.asarray(vessel_amplitudes),
    )
    if any(len(value) != expected for value in arrays):
        raise ValueError(f"every continuous input must have {expected} frames")
    reference = PADDING_BEATS * FRAMES_PER_BEAT
    keep = retained_frame_indices()
    sigma = referenced_hp_lp(
        arrays[0], window=movmean_window, time_axis=0, reference_index=reference
    )
    voltage = referenced_hp_lp(
        arrays[1], window=movmean_window, time_axis=0, reference_index=reference
    )
    clean = referenced_hp_lp(
        arrays[2], window=movmean_window, time_axis=0, reference_index=reference
    )
    amplitudes = referenced_hp_lp(
        arrays[3], window=movmean_window, time_axis=0, reference_index=reference
    )
    np.testing.assert_allclose(sigma.hp + sigma.lp, sigma.full, atol=2e-10, rtol=2e-7)
    np.testing.assert_allclose(
        voltage.hp + voltage.lp, voltage.full, atol=2e-12, rtol=2e-7
    )
    output = {}
    full_sigma = sigma.full[keep]
    sigma_reference = np.broadcast_to(
        arrays[0][reference][None, :], full_sigma.shape
    )
    np.testing.assert_allclose(
        sigma_reference + full_sigma,
        arrays[0][keep],
        atol=2e-10,
        rtol=2e-7,
    )
    full_voltage = voltage.full[keep]
    voltage_reference = np.broadcast_to(
        arrays[1][reference][None, :], full_voltage.shape
    )
    np.testing.assert_allclose(
        voltage_reference + full_voltage,
        arrays[1][keep],
        atol=2e-12,
        rtol=2e-7,
    )
    absolute_sigma = arrays[0][keep]
    absolute_voltage = arrays[1][keep]
    clean_absolute_voltage = arrays[2][keep]
    output["full"] = {
        # These are the canonical one-frame training pair for the full-band
        # GCNM.  They are deliberately absolute, not referenced deltas.
        "sigma": absolute_sigma,
        "V": absolute_voltage,
        "V_clean": clean_absolute_voltage,
        "V_template": _frame_beat_templates(absolute_voltage),
        # Keep explicit aliases and referenced differences so the absolute
        # identity and the post-reconstruction HP/LP decomposition are both
        # auditable without overloading the training fields.
        "sigma_absolute": absolute_sigma,
        "sigma_delta_reference": full_sigma,
        "sigma_reference": sigma_reference,
        "V_delta_reference": full_voltage,
        "V_clean_delta_reference": clean.full[keep],
        "V_absolute_filtered": absolute_voltage,
        "V_clean_absolute_filtered": clean_absolute_voltage,
        "V_reference_filtered": voltage_reference,
        "vessel_amplitudes": amplitudes.full[keep],
    }
    if sigma_resting is not None:
        resting = np.asarray(sigma_resting)
        if resting.shape != arrays[0].shape[1:]:
            raise ValueError("sigma_resting must match one absolute conductivity frame")
        output["full"]["sigma_resting"] = np.broadcast_to(
            resting[None, :], full_sigma.shape
        )
    if voltage_raw is not None or voltage_clean_raw is not None:
        if voltage_raw is None or voltage_clean_raw is None:
            raise ValueError("raw augmented and clean voltages must be supplied together")
        raw = np.asarray(voltage_raw)
        raw_clean = np.asarray(voltage_clean_raw)
        if len(raw) != expected or raw.shape != raw_clean.shape:
            raise ValueError("raw voltage arrays must have matching continuous shapes")
        output["full"]["V_absolute_raw"] = raw[keep]
        output["full"]["V_clean_absolute_raw"] = raw_clean[keep]
    for name in ("hp", "lp"):
        component_voltage = getattr(voltage, name)[keep]
        output[name] = {
            "sigma": getattr(sigma, name)[keep],
            "V": component_voltage,
            "V_clean": getattr(clean, name)[keep],
            "V_template": _frame_beat_templates(component_voltage),
            "vessel_amplitudes": getattr(amplitudes, name)[keep],
        }
    return output


def _smooth_curve(
    rng: np.random.Generator,
    length: int,
    *,
    anchors: int,
    peak: float = 1.0,
) -> np.ndarray:
    control = rng.normal(size=anchors)
    control -= np.mean(control)
    curve = np.interp(
        np.linspace(0.0, 1.0, length),
        np.linspace(0.0, 1.0, anchors),
        control,
    )
    scale = max(float(np.max(np.abs(curve))), 1e-12)
    return peak * curve / scale


def _slow_nonvascular_field(
    points: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    *,
    frames: int,
    maximum_amplitude: float,
) -> np.ndarray:
    output = np.zeros((frames, len(points)), dtype=np.float64)
    muscle = np.flatnonzero(labels == 3)
    if not len(muscle) or maximum_amplitude <= 0:
        return output
    for _ in range(int(rng.integers(1, 4))):
        center = points[int(rng.choice(muscle))]
        width_x = float(rng.uniform(1.5, 4.5))
        width_y = float(rng.uniform(1.5, 4.5))
        angle = float(rng.uniform(-np.pi, np.pi))
        cosine, sine = np.cos(angle), np.sin(angle)
        offset = points - center[None, :]
        local_x = cosine * offset[:, 0] + sine * offset[:, 1]
        local_y = -sine * offset[:, 0] + cosine * offset[:, 1]
        spatial = np.exp(
            -0.5 * ((local_x / width_x) ** 2 + (local_y / width_y) ** 2)
        )
        spatial[labels != 3] = 0.0
        amplitude = float(rng.uniform(0.2, 1.0) * maximum_amplitude)
        temporal = _smooth_curve(rng, frames, anchors=7, peak=amplitude)
        output += temporal[:, None] * spatial[None, :]
    return output


def _measurement_effects(
    clean_absolute: np.ndarray,
    rng: np.random.Generator,
    noise: dict[str, float],
) -> tuple[np.ndarray, dict]:
    """Apply coherent acquisition variation while retaining signed channels."""

    clean = np.asarray(clean_absolute, dtype=np.float64)
    frames, channels = clean.shape
    channel_gain = np.exp(rng.normal(0.0, noise["channel_gain_sd"], channels))
    current_gain = float(np.exp(rng.normal(0.0, noise["current_gain_sd"])))
    output = current_gain * clean * channel_gain[None, :]
    absolute_rms = np.sqrt(np.mean(clean * clean, axis=0))
    dynamic_rms = np.std(clean, axis=0)
    scale = np.maximum(dynamic_rms, float(noise["noise_floor"]))

    # A static offset is present in absolute measurements; it disappears only
    # after the explicit component reference.  Contact drift does not.
    offset = rng.normal(0.0, noise["offset_rel"], channels) * absolute_rms
    output += offset[None, :]
    contact_curve = _smooth_curve(rng, frames, anchors=9, peak=1.0)
    contact_spatial = rng.normal(size=channels)
    output += noise["contact_drift_sd"] * contact_curve[:, None] * absolute_rms[
        None, :
    ] * contact_spatial[None, :]
    output += rng.normal(size=output.shape) * (
        noise["white_noise_rel"] * scale[None, :] + noise["noise_floor"]
    )
    spatial = rng.normal(size=channels)
    spatial /= max(float(np.linalg.norm(spatial)), 1e-12)
    correlated = _smooth_curve(rng, frames, anchors=13, peak=1.0)
    output += (
        noise["correlated_noise_rel"]
        * float(np.sqrt(np.mean(scale * scale)))
        * np.sqrt(channels)
        * correlated[:, None]
        * spatial[None, :]
    )
    artifact_events = []
    if rng.random() < noise["artifact_probability"]:
        count = int(rng.integers(1, 4))
        for _ in range(count):
            frame = int(rng.integers(0, frames))
            selected = rng.choice(channels, size=int(rng.integers(1, 5)), replace=False)
            increments = rng.normal(
                0.0,
                noise["artifact_scale"] * max(float(np.max(scale)), 1e-12),
                len(selected),
            )
            output[frame, selected] += increments
            artifact_events.append(
                {
                    "frame": frame,
                    "channels": selected.astype(int).tolist(),
                    "increments_v": increments.tolist(),
                }
            )
    return output, {
        "current_gain": current_gain,
        "channel_gain": channel_gain.tolist(),
        "static_offset_v": offset.tolist(),
        "contact_drift_spatial": contact_spatial.tolist(),
        "correlated_noise_spatial": spatial.tolist(),
        "artifact_events": artifact_events,
        "note": (
            "contact drift is a voltage-domain proxy; contact impedance and "
            "electrode geometry are not resampled in the FEM"
        ),
    }


def _build_absolute_jacobian_bank(
    *,
    count: int,
    rng: np.random.Generator,
    augmentation_spec,
    baseline_model,
    simulator: dict,
    inverse_mesh,
    forward_mesh,
    physics,
) -> list[dict[str, np.ndarray]]:
    bank = []
    for index in range(count):
        model = simulator["augment_model"](baseline_model, augmentation_spec, rng)
        inverse = simulator["simulate_points"](
            simulator["mesh_points_mm"](inverse_mesh, model),
            model,
            simulator["WaveformSpec"](frames=FRAMES_PER_BEAT),
        )
        forward = simulator["simulate_points"](
            simulator["mesh_points_mm"](forward_mesh, model),
            model,
            simulator["WaveformSpec"](frames=FRAMES_PER_BEAT),
        )
        skin = float(model.conductivities.skin_s_m)
        baseline_inv, _ = _fill_outside(inverse, skin)
        baseline_fwd, _ = _fill_outside(forward, skin)
        forward_state = physics._forward(baseline_fwd)
        bank.append(
            {
                "baseline_inv": baseline_inv,
                "baseline_fwd": baseline_fwd,
                "voltage": np.real(
                    np.asarray(forward_state.results.vmeas, dtype=np.complex128)
                ).ravel(),
                "jacobian": np.asarray(
                    physics.jacobian_from_forward(forward_state), dtype=np.float64
                ),
            }
        )
        print(f"built absolute fine-mesh Jacobian {index + 1}/{count}", flush=True)
    return bank


def _exact_voltage_sequence(physics, sigma_absolute: np.ndarray) -> np.ndarray:
    output = []
    for index, sigma in enumerate(sigma_absolute):
        output.append(np.real(np.asarray(physics.solve(sigma), dtype=np.complex128)))
        if (index + 1) % 50 == 0:
            print(f"  exact forward frames {index + 1}/{len(sigma_absolute)}", flush=True)
    return np.stack(output)


def _linearized_voltage_sequence(
    bank: list[dict[str, np.ndarray]],
    baseline_inv: np.ndarray,
    sigma_absolute_fwd: np.ndarray,
) -> tuple[np.ndarray, int]:
    distances = [
        float(np.mean((entry["baseline_inv"] - baseline_inv) ** 2)) for entry in bank
    ]
    index = int(np.argmin(distances))
    selected = bank[index]
    delta = sigma_absolute_fwd - selected["baseline_fwd"][None, :]
    return selected["voltage"][None, :] + delta @ selected["jacobian"].T, index


def _anatomy_linearized_voltage_sequence(
    physics,
    baseline_fwd: np.ndarray,
    sigma_absolute_fwd: np.ndarray,
) -> np.ndarray:
    """Linearize all frames at the exact resting state of this anatomy.

    A nearest-neighbour bank can have a large absolute tissue/geometry
    mismatch.  One forward/Jacobian solve per virtual anatomy removes that
    mismatch while retaining the roughly 350x acceleration over an exact
    solve at every frame.
    """

    forward = physics._forward(np.asarray(baseline_fwd, dtype=np.float64))
    voltage = np.real(
        np.asarray(forward.results.vmeas, dtype=np.complex128)
    ).ravel()
    jacobian = np.asarray(physics.jacobian_from_forward(forward), dtype=np.float64)
    delta = np.asarray(sigma_absolute_fwd, dtype=np.float64) - np.asarray(
        baseline_fwd, dtype=np.float64
    )[None, :]
    return voltage[None, :] + delta @ jacobian.T


def _continuous_anatomy(
    *,
    anatomy_id: int,
    rng: np.random.Generator,
    baseline_model,
    baseline_waveform,
    augmentation_spec,
    simulator: dict,
    inverse_mesh,
    forward_mesh,
    domain_center: np.ndarray,
    domain_radius: float,
    physics,
    jacobian_bank: list[dict[str, np.ndarray]],
    linearization_reference: str,
    mode: str,
    noise: dict[str, float],
    finger_position_mm: float,
    slow_nonvascular_amplitude: float,
    slow_vascular_fraction: float,
    sampling_rate_hz: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    model = simulator["augment_model"](baseline_model, augmentation_spec, rng)
    physical_offset = rng.uniform(-finger_position_mm, finger_position_mm, size=2)
    inverse_points = (
        simulator["mesh_points_mm"](inverse_mesh, model) - physical_offset[None, :]
    )
    forward_points = (
        simulator["mesh_points_mm"](forward_mesh, model) - physical_offset[None, :]
    )

    inverse_delta_parts = []
    forward_delta_parts = []
    amplitude_parts = []
    durations = []
    baseline_inv = baseline_fwd = labels_inv = labels_fwd = None
    first_waveform = None
    beat_waveforms = []
    for beat in range(TOTAL_BEATS):
        waveform_spec = simulator["augment_waveform"](
            baseline_waveform, augmentation_spec, rng
        )
        inverse = simulator["simulate_points"](inverse_points, model, waveform_spec)
        forward = simulator["simulate_points"](forward_points, model, waveform_spec)
        if first_waveform is None:
            first_waveform = np.asarray(inverse.waveform)
        beat_waveforms.append(np.asarray(inverse.waveform).tolist())
        if not np.allclose(inverse.waveform, forward.waveform):
            raise RuntimeError("inverse and forward beat waveforms differ")
        skin = float(model.conductivities.skin_s_m)
        current_baseline_inv, delta_inv = _fill_outside(inverse, skin)
        current_baseline_fwd, delta_fwd = _fill_outside(forward, skin)
        if baseline_inv is None:
            baseline_inv = current_baseline_inv
            baseline_fwd = current_baseline_fwd
            labels_inv = np.asarray(inverse.tissue_labels, dtype=np.uint8)
            labels_fwd = np.asarray(forward.tissue_labels, dtype=np.uint8)
        else:
            np.testing.assert_allclose(current_baseline_inv, baseline_inv)
            np.testing.assert_allclose(current_baseline_fwd, baseline_fwd)
        beat_scale = float(rng.uniform(0.70, 1.30))
        inverse_delta_parts.append(delta_inv * beat_scale)
        forward_delta_parts.append(delta_fwd * beat_scale)
        amplitudes = []
        for artery in model.arteries:
            shifted = simulator["phase_shift"](
                np.asarray(inverse.waveform), float(artery.phase_delay_fraction)
            )
            amplitudes.append(
                artery.peak_delta_s_m * artery.waveform_scale * beat_scale * shifted
            )
        amplitude_parts.append(np.column_stack(amplitudes))
        durations.append(float(waveform_spec.duration_s))

    assert baseline_inv is not None and baseline_fwd is not None
    assert labels_inv is not None and labels_fwd is not None
    delta_inv = np.concatenate(inverse_delta_parts, axis=0)
    delta_fwd = np.concatenate(forward_delta_parts, axis=0)
    vessel_amplitudes = np.concatenate(amplitude_parts, axis=0)
    frames = len(delta_inv)

    slow_curve = _smooth_curve(rng, frames, anchors=8, peak=slow_vascular_fraction)
    peak_index = int(np.argmax(np.sqrt(np.mean(delta_inv * delta_inv, axis=1))))
    vascular_inv = delta_inv[peak_index]
    vascular_fwd = delta_fwd[peak_index]
    delta_inv += slow_curve[:, None] * vascular_inv[None, :]
    delta_fwd += slow_curve[:, None] * vascular_fwd[None, :]
    peak_amplitudes = np.max(np.abs(vessel_amplitudes), axis=0)
    vessel_amplitudes += slow_curve[:, None] * peak_amplitudes[None, :]

    delta_inv += _slow_nonvascular_field(
        inverse_points,
        labels_inv,
        rng,
        frames=frames,
        maximum_amplitude=slow_nonvascular_amplitude,
    )
    delta_fwd += _slow_nonvascular_field(
        forward_points,
        labels_fwd,
        rng,
        frames=frames,
        maximum_amplitude=slow_nonvascular_amplitude,
    )
    sigma_inv = np.maximum(baseline_inv[None, :] + delta_inv, 1e-5)
    sigma_fwd = np.maximum(baseline_fwd[None, :] + delta_fwd, 1e-5)

    bank_index = None
    linearized_absolute = None
    if linearization_reference == "anatomy":
        linearized_absolute = _anatomy_linearized_voltage_sequence(
            physics, baseline_fwd, sigma_fwd
        )
        bank_index = -1
    elif linearization_reference == "bank":
        if not jacobian_bank:
            raise ValueError("bank linearization requires a Jacobian bank")
        linearized_absolute, bank_index = _linearized_voltage_sequence(
            jacobian_bank, baseline_inv, sigma_fwd
        )
    else:
        raise ValueError(
            f"unsupported linearization reference {linearization_reference!r}"
        )

    if mode == "nonlinear":
        voltage_clean_absolute = _exact_voltage_sequence(physics, sigma_fwd)
    elif mode == "linearized":
        voltage_clean_absolute = linearized_absolute
    else:
        raise ValueError(f"unsupported simulation mode {mode}")
    voltage_absolute, acquisition_effects = _measurement_effects(
        voltage_clean_absolute, rng, noise
    )
    voltage_filtered = butterworth_lowpass_5hz(
        voltage_absolute, sampling_rate_hz=sampling_rate_hz, axis=0
    )
    voltage_clean_filtered = butterworth_lowpass_5hz(
        voltage_clean_absolute, sampling_rate_hz=sampling_rate_hz, axis=0
    )
    components = decompose_and_retain(
        sigma_inv,
        voltage_filtered,
        voltage_clean_filtered,
        vessel_amplitudes,
        sigma_resting=baseline_inv,
        voltage_raw=voltage_absolute,
        voltage_clean_raw=voltage_clean_absolute,
    )
    approximation = None
    if linearized_absolute is not None:
        linearized_filtered = butterworth_lowpass_5hz(
            linearized_absolute, sampling_rate_hz=sampling_rate_hz, axis=0
        )
        linearized_components = referenced_hp_lp(
            linearized_filtered,
            window=PRODUCTION_MOVMEAN_WINDOW,
            time_axis=0,
            reference_index=PADDING_BEATS * FRAMES_PER_BEAT,
        )
        keep = retained_frame_indices()
        approximation = {}
        for component in ("full", "hp", "lp"):
            estimate = (
                linearized_filtered[keep]
                if component == "full"
                else getattr(linearized_components, component)[keep]
            )
            exact = components[component]["V_clean"]
            residual = estimate - exact
            exact_rms = max(float(np.sqrt(np.mean(exact * exact))), 1e-12)
            correlation = float(
                np.corrcoef(estimate.reshape(-1), exact.reshape(-1))[0, 1]
            )
            approximation[component] = {
                "nrmse": float(np.sqrt(np.mean(residual * residual)) / exact_rms),
                "correlation": correlation,
            }
    geometry, _ = _shifted_vessel_record(
        model,
        np.asarray(first_waveform),
        0,
        inverse_mesh,
        domain_center,
        domain_radius,
        simulator["phase_shift"],
        physical_offset,
    )
    return components, {
        "anatomy_id": int(anatomy_id),
        "model": model.to_dict(),
        "tissue_labels": labels_inv,
        "vessels": geometry,
        "finger_offset_in_ring_mm": physical_offset.tolist(),
        "beat_durations_s": durations,
        "beat_waveforms": beat_waveforms,
        "acquisition_effects": acquisition_effects,
        "simulation_mode": mode,
        "jacobian_bank_index": bank_index,
        "linearized_vs_exact": approximation,
    }


def _empty_component_arrays(samples: int, elements: int, measurements: int) -> dict:
    return {
        "sigma": np.empty((samples, elements), dtype=np.float32),
        "sigma_baseline": np.full((samples, elements), 0.7, dtype=np.float32),
        "V": np.empty((samples, measurements), dtype=np.float32),
        "V_clean": np.empty((samples, measurements), dtype=np.float32),
        "V_template": np.empty((samples, measurements), dtype=np.float32),
        "tissue_labels": np.empty((samples, elements), dtype=np.uint8),
        "anatomy_id": np.empty(samples, dtype=np.int32),
        "beat_id": np.empty(samples, dtype=np.int32),
        "sample_index": np.empty(samples, dtype=np.int16),
        "phase_index": np.empty(samples, dtype=np.int16),
    }


def _empty_full_arrays(samples: int, elements: int, measurements: int) -> dict:
    arrays = _empty_component_arrays(samples, elements, measurements)
    arrays.update(
        {
            "sigma_absolute": np.empty((samples, elements), dtype=np.float32),
            "sigma_delta_reference": np.empty(
                (samples, elements), dtype=np.float32
            ),
            "sigma_reference": np.empty((samples, elements), dtype=np.float32),
            "sigma_resting": np.empty((samples, elements), dtype=np.float32),
            "V_absolute_raw": np.empty((samples, measurements), dtype=np.float32),
            "V_clean_absolute_raw": np.empty(
                (samples, measurements), dtype=np.float32
            ),
            "V_absolute_filtered": np.empty(
                (samples, measurements), dtype=np.float32
            ),
            "V_clean_absolute_filtered": np.empty(
                (samples, measurements), dtype=np.float32
            ),
            "V_reference_filtered": np.empty(
                (samples, measurements), dtype=np.float32
            ),
            "V_delta_reference": np.empty(
                (samples, measurements), dtype=np.float32
            ),
            "V_clean_delta_reference": np.empty(
                (samples, measurements), dtype=np.float32
            ),
        }
    )
    return arrays


def _production_newton_control(
    inverse_matrix: np.ndarray, voltage: np.ndarray
) -> np.ndarray:
    """Apply the already-signed production inverse matrix to row-wise ΔV."""

    return (
        np.asarray(inverse_matrix, dtype=np.float64)
        @ np.asarray(voltage, dtype=np.float64).T
    ).T.astype(np.float32)


def _component_records(anatomy: dict, amplitudes: np.ndarray) -> list[dict]:
    records = []
    for frame, values in enumerate(amplitudes):
        retained_beat = frame // FRAMES_PER_BEAT
        sample = frame % FRAMES_PER_BEAT
        records.append(
            {
                "rotation": 0.0,
                "vessels": anatomy["vessels"],
                "vessel_delta": [float(value) for value in values],
                "anatomy_id": anatomy["anatomy_id"],
                "beat_id": anatomy["anatomy_id"] * RETAINED_BEATS + retained_beat,
                "sample_index": sample,
            }
        )
    return records


def _generate_split(
    spec: SplitSpec,
    *,
    baseline_model,
    baseline_waveform,
    augmentation_spec,
    simulator: dict,
    inverse_mesh,
    forward_mesh,
    domain_center: np.ndarray,
    domain_radius: float,
    physics,
    jacobian_bank: list[dict[str, np.ndarray]],
    linearization_reference: str,
    inverse_matrix: np.ndarray,
    noise: dict[str, float],
    finger_position_mm: float,
    slow_nonvascular_amplitude: float,
    slow_vascular_fraction: float,
    sampling_rate_hz: float,
    output_root: Path,
) -> dict:
    rng = np.random.default_rng(spec.seed)
    count = len(spec.anatomy_ids) * RETAINED_BEATS * FRAMES_PER_BEAT
    elements = inverse_matrix.shape[0]
    measurements = inverse_matrix.shape[1]
    arrays = {
        component: _empty_component_arrays(count, elements, measurements)
        for component in ("hp", "lp")
    }
    arrays["full"] = _empty_full_arrays(count, elements, measurements)
    records: dict[str, list[dict]] = {"hp": [], "lp": []}
    anatomy_records: list[dict] = []
    cursor = 0
    component_rms: dict[str, list[float]] = {name: [] for name in ("full", "hp", "lp")}
    approximation_metrics: dict[str, list[dict]] = {
        name: [] for name in ("full", "hp", "lp")
    }
    for local_index, anatomy_id in enumerate(spec.anatomy_ids):
        components, anatomy = _continuous_anatomy(
            anatomy_id=int(anatomy_id),
            rng=rng,
            baseline_model=baseline_model,
            baseline_waveform=baseline_waveform,
            augmentation_spec=augmentation_spec,
            simulator=simulator,
            inverse_mesh=inverse_mesh,
            forward_mesh=forward_mesh,
            domain_center=domain_center,
            domain_radius=domain_radius,
            physics=physics,
            jacobian_bank=jacobian_bank,
            linearization_reference=linearization_reference,
            mode=spec.mode,
            noise=noise,
            finger_position_mm=finger_position_mm,
            slow_nonvascular_amplitude=slow_nonvascular_amplitude,
            slow_vascular_fraction=slow_vascular_fraction,
            sampling_rate_hz=sampling_rate_hz,
        )
        stop = cursor + RETAINED_BEATS * FRAMES_PER_BEAT
        if anatomy["linearized_vs_exact"] is not None:
            for component in ("full", "hp", "lp"):
                approximation_metrics[component].append(
                    anatomy["linearized_vs_exact"][component]
                )
        labels = np.broadcast_to(
            anatomy["tissue_labels"][None, :],
            (RETAINED_BEATS * FRAMES_PER_BEAT, elements),
        )
        beat_ids = np.repeat(
            int(anatomy_id) * RETAINED_BEATS + np.arange(RETAINED_BEATS),
            FRAMES_PER_BEAT,
        )
        sample_indices = np.tile(np.arange(FRAMES_PER_BEAT), RETAINED_BEATS)
        for component in ("full", "hp", "lp"):
            values = components[component]
            arrays[component]["sigma"][cursor:stop] = values["sigma"]
            arrays[component]["V"][cursor:stop] = values["V"]
            arrays[component]["V_clean"][cursor:stop] = values["V_clean"]
            arrays[component]["V_template"][cursor:stop] = values["V_template"]
            arrays[component]["tissue_labels"][cursor:stop] = labels
            arrays[component]["anatomy_id"][cursor:stop] = int(anatomy_id)
            arrays[component]["beat_id"][cursor:stop] = beat_ids
            arrays[component]["sample_index"][cursor:stop] = sample_indices
            arrays[component]["phase_index"][cursor:stop] = sample_indices
            if component in records:
                records[component].extend(
                    _component_records(anatomy, values["vessel_amplitudes"])
                )
            component_rms[component].append(
                float(np.sqrt(np.mean(values["sigma"] ** 2)))
            )
        full_values = components["full"]
        for key in (
            "sigma_absolute",
            "sigma_delta_reference",
            "sigma_reference",
            "sigma_resting",
            "V_absolute_raw",
            "V_clean_absolute_raw",
            "V_absolute_filtered",
            "V_clean_absolute_filtered",
            "V_reference_filtered",
            "V_delta_reference",
            "V_clean_delta_reference",
        ):
            arrays["full"][key][cursor:stop] = full_values[key]
        anatomy_records.append(
            {key: value for key, value in anatomy.items() if key != "tissue_labels"}
        )
        cursor = stop
        print(
            f"{spec.name}: generated anatomy {local_index + 1}/{len(spec.anatomy_ids)} "
            f"(id={int(anatomy_id)}, {spec.mode})",
            flush=True,
        )
    if cursor != count:
        raise RuntimeError(f"filled {cursor} samples, expected {count}")

    output = {}
    for component in ("full", "hp", "lp"):
        component_root = output_root / component
        component_root.mkdir(parents=True, exist_ok=True)
        if spec.name == "test":
            # production_reconstruction_matrix already includes MATLAB's
            # leading minus sign: Δσ = (-D1) @ ΔV.
            arrays[component]["newton"] = _production_newton_control(
                inverse_matrix,
                (
                    arrays[component]["V_delta_reference"]
                    if component == "full"
                    else arrays[component]["V"]
                ),
            )
        np.savez_compressed(component_root / f"{spec.name}.npz", **arrays[component])
        if component in records:
            (component_root / f"{spec.name}_anatomy.json").write_text(
                json.dumps(records[component]), encoding="utf-8"
            )
        else:
            (component_root / f"{spec.name}_anatomies.json").write_text(
                json.dumps(anatomy_records), encoding="utf-8"
            )
        output[component] = {
            "samples": count,
            "retained_beats": len(spec.anatomy_ids) * RETAINED_BEATS,
            "anatomies": len(spec.anatomy_ids),
            "activation_rms_minimum_by_anatomy": min(component_rms[component]),
            "negative_target_fraction": float(np.mean(arrays[component]["sigma"] < 0)),
            "voltage_rms": float(np.sqrt(np.mean(arrays[component]["V"] ** 2))),
            "absolute_identity_max_error": (
                float(
                    np.max(
                        np.abs(
                            arrays["full"]["sigma_reference"]
                            + arrays["full"]["sigma_delta_reference"]
                            - arrays["full"]["sigma"]
                        )
                    )
                )
                if component == "full"
                else None
            ),
            "linearized_vs_exact": (
                {
                    "anatomies": len(approximation_metrics[component]),
                    "nrmse_mean": float(
                        np.mean(
                            [item["nrmse"] for item in approximation_metrics[component]]
                        )
                    ),
                    "nrmse_maximum": float(
                        np.max(
                            [item["nrmse"] for item in approximation_metrics[component]]
                        )
                    ),
                    "correlation_mean": float(
                        np.mean(
                            [
                                item["correlation"]
                                for item in approximation_metrics[component]
                            ]
                        )
                    ),
                }
                if approximation_metrics[component]
                else None
            ),
        }
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulator-root",
        type=Path,
        default=root.parent / "Finger-Conductivity-Simulator",
    )
    parser.add_argument("--finger-config", type=Path)
    parser.add_argument(
        "--config", type=Path, default=root / "configs/rings_b045/US120.yaml"
    )
    parser.add_argument(
        "--output-root", type=Path, default=root / "data/full_band_beats_US120_v2"
    )
    parser.add_argument("--train-anatomies", type=int, default=160)
    parser.add_argument("--validation-anatomies", type=int, default=20)
    parser.add_argument("--test-anatomies", type=int, default=20)
    parser.add_argument(
        "--train-mode", choices=["linearized", "nonlinear"], default="linearized"
    )
    parser.add_argument(
        "--validation-mode", choices=["linearized", "nonlinear"], default="nonlinear"
    )
    parser.add_argument(
        "--test-mode", choices=["linearized", "nonlinear"], default="nonlinear"
    )
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--jacobian-bank-size", type=int, default=16)
    parser.add_argument(
        "--linearization-reference",
        choices=["anatomy", "bank"],
        default="anatomy",
        help=(
            "use one exact resting Jacobian per anatomy (default), or the "
            "legacy nearest-anatomy bank"
        ),
    )
    parser.add_argument(
        "--maximum-linearized-component-nrmse",
        type=float,
        default=0.50,
        help="hard acceptance limit measured on exact nonlinear validation/test anatomies",
    )
    parser.add_argument(
        "--minimum-linearized-component-correlation",
        type=float,
        default=0.80,
        help="hard acceptance limit measured on exact nonlinear validation/test anatomies",
    )
    parser.add_argument("--sampling-rate-hz", type=float, default=50.0)
    parser.add_argument("--finger-size-variation", type=float, default=0.15)
    parser.add_argument("--finger-rotation-deg", type=float, default=20.0)
    parser.add_argument("--finger-position-mm", type=float, default=1.0)
    parser.add_argument("--artery-size-variation", type=float, default=0.35)
    parser.add_argument("--artery-position-mm", type=float, default=1.8)
    parser.add_argument("--artery-rotation-deg", type=float, default=30.0)
    parser.add_argument("--conductivity-variation", type=float, default=0.20)
    parser.add_argument("--diffusion-variation", type=float, default=0.75)
    parser.add_argument("--waveform-shape-variation", type=float, default=0.20)
    parser.add_argument("--duration-variation", type=float, default=0.15)
    parser.add_argument("--slow-nonvascular-amplitude", type=float, default=0.006)
    parser.add_argument("--slow-vascular-fraction", type=float, default=0.30)
    parser.add_argument("--white-noise-rel", type=float, default=0.04)
    parser.add_argument("--correlated-noise-rel", type=float, default=0.04)
    parser.add_argument("--noise-floor", type=float, default=1e-7)
    parser.add_argument("--contact-drift-sd", type=float, default=0.002)
    parser.add_argument("--channel-gain-sd", type=float, default=0.015)
    parser.add_argument("--current-gain-sd", type=float, default=0.03)
    parser.add_argument("--offset-rel", type=float, default=0.01)
    parser.add_argument("--artifact-probability", type=float, default=0.03)
    parser.add_argument("--artifact-scale", type=float, default=0.25)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"immutable synthetic root already exists: {args.output_root}")
    if args.sampling_rate_hz <= 10.0:
        raise ValueError("sampling rate must exceed twice the production 5 Hz cutoff")
    splits = split_anatomy_ids(
        args.train_anatomies,
        args.validation_anatomies,
        args.test_anatomies,
        seed=args.seed,
    )
    requested_modes = {
        "train": args.train_mode,
        "validation": args.validation_mode,
        "test": args.test_mode,
    }
    splits = tuple(
        SplitSpec(spec.name, spec.anatomy_ids, requested_modes[spec.name], spec.seed)
        for spec in splits
    )
    sets = [set(spec.anatomy_ids.tolist()) for spec in splits]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("virtual anatomy leakage across splits")

    simulator = _load_simulator(args.simulator_root.resolve())
    finger_config = args.finger_config or (
        args.simulator_root / "configs/default_finger.json"
    )
    baseline_model = simulator["FingerModel"].from_dict(
        json.loads(finger_config.read_text(encoding="utf-8"))
    )
    baseline_waveform = simulator["WaveformSpec"](
        kind="heartbeat", frames=FRAMES_PER_BEAT, duration_s=1.0
    )
    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=True)
    inverse_matrix = production_reconstruction_matrix(
        runtime["physics_inv"],
        hyper_pvi=cfg.hyper_pvi,
        regularizer=runtime["mappings"].laplace,
    )
    inverse_mesh = simulator["load_ring_mesh"](Path(cfg.mesh_inv_h5), "hp_lp_inverse")
    forward_mesh = simulator["load_ring_mesh"](Path(cfg.mesh_fwd_h5), "hp_lp_forward")
    center, radius = domain_transform(runtime["mesh_inv"])
    augmentation = {
        "finger_size_fraction": args.finger_size_variation,
        "finger_rotation_deg": args.finger_rotation_deg,
        "artery_size_fraction": args.artery_size_variation,
        "artery_position_mm": args.artery_position_mm,
        "artery_rotation_deg": args.artery_rotation_deg,
        "conductivity_fraction": args.conductivity_variation,
        "diffusion_fraction": args.diffusion_variation,
        "waveform_shape_fraction": args.waveform_shape_variation,
        "duration_fraction": args.duration_variation,
    }
    augmentation_spec = simulator["AugmentationSpec"](
        samples=sum(len(spec.anatomy_ids) for spec in splits),
        seed=args.seed,
        **augmentation,
    )
    augmentation_spec.validate()
    noise = {
        "white_noise_rel": args.white_noise_rel,
        "correlated_noise_rel": args.correlated_noise_rel,
        "noise_floor": args.noise_floor,
        "contact_drift_sd": args.contact_drift_sd,
        "channel_gain_sd": args.channel_gain_sd,
        "current_gain_sd": args.current_gain_sd,
        "offset_rel": args.offset_rel,
        "artifact_probability": args.artifact_probability,
        "artifact_scale": args.artifact_scale,
    }
    bank = (
        _build_absolute_jacobian_bank(
            count=args.jacobian_bank_size,
            rng=np.random.default_rng(args.seed + 11),
            augmentation_spec=augmentation_spec,
            baseline_model=baseline_model,
            simulator=simulator,
            inverse_mesh=inverse_mesh,
            forward_mesh=forward_mesh,
            physics=runtime["physics_fwd"],
        )
        if args.linearization_reference == "bank"
        else []
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    incomplete = args.output_root / "_INCOMPLETE"
    incomplete.write_text("synthetic generation in progress\n", encoding="utf-8")
    split_reports = {}
    for spec in splits:
        split_reports[spec.name] = _generate_split(
            spec,
            baseline_model=baseline_model,
            baseline_waveform=baseline_waveform,
            augmentation_spec=augmentation_spec,
            simulator=simulator,
            inverse_mesh=inverse_mesh,
            forward_mesh=forward_mesh,
            domain_center=center,
            domain_radius=radius,
            physics=runtime["physics_fwd"],
            jacobian_bank=bank,
            linearization_reference=args.linearization_reference,
            inverse_matrix=inverse_matrix,
            noise=noise,
            finger_position_mm=args.finger_position_mm,
            slow_nonvascular_amplitude=args.slow_nonvascular_amplitude,
            slow_vascular_fraction=args.slow_vascular_fraction,
            sampling_rate_hz=args.sampling_rate_hz,
            output_root=args.output_root,
        )

    exact_comparisons = []
    for split in ("validation", "test"):
        for component in ("full", "hp", "lp"):
            comparison = split_reports[split][component]["linearized_vs_exact"]
            if comparison is not None:
                exact_comparisons.append((split, component, comparison))
    if not exact_comparisons:
        print(
            "warning: no exact-nonlinear split was requested; acceleration gate is unverified",
            flush=True,
        )
    for split, component, comparison in exact_comparisons:
        if comparison["nrmse_maximum"] > args.maximum_linearized_component_nrmse:
            raise RuntimeError(
                f"{split}/{component} linearized NRMSE "
                f"{comparison['nrmse_maximum']:.4f} exceeds "
                f"{args.maximum_linearized_component_nrmse:.4f}"
            )
        if comparison["correlation_mean"] < args.minimum_linearized_component_correlation:
            raise RuntimeError(
                f"{split}/{component} linearized correlation "
                f"{comparison['correlation_mean']:.4f} is below "
                f"{args.minimum_linearized_component_correlation:.4f}"
            )

    metadata = {
        "schema": "pvi-gcnm-continuous-full-hp-lp-beats-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ring": Path(args.config).stem,
        "components": ["full", "hp", "lp"],
        "stored_conductivity_contract": {
            "sigma_resting": "anatomy-specific resting conductivity on inverse mesh",
            "sigma_reference": "true absolute conductivity at first retained frame",
            "sigma_absolute": "true retained absolute conductivity frame",
            "sigma": "canonical full-band GCNM target: true absolute conductivity",
            "sigma_delta_reference": "full conductivity minus first retained frame",
            "identity": "sigma = sigma_absolute = sigma_reference + sigma_delta_reference",
        },
        "stored_voltage_contract": {
            "V_absolute_raw": "augmented absolute forward voltage before 5 Hz filtering",
            "V_clean_absolute_raw": "clean absolute forward voltage before filtering",
            "V_absolute_filtered": "augmented absolute voltage after 5 Hz filtering",
            "V_clean_absolute_filtered": "clean absolute voltage after filtering",
            "V_reference_filtered": "filtered augmented voltage at first retained frame",
            "V": "canonical full-band GCNM input: augmented absolute filtered voltage",
            "V_clean": "clean absolute filtered voltage",
            "V_delta_reference": "filtered voltage minus first retained frame",
            "V_clean_delta_reference": "clean filtered voltage minus its reference",
        },
        "full_band_training_contract": "one frame: V_absolute_filtered -> sigma_absolute",
        "component_derivation_contract": (
            "apply the identical 100-frame moving-mean decomposition to the "
            "predicted absolute conductivity sequence after reconstruction"
        ),
        "reference_rule": "HP and LP sum to the full difference from the first retained frame",
        "temporal_processing": {
            "voltage_filter": "third-order zero-phase Butterworth low-pass 5 Hz",
            "sampling_rate_hz": args.sampling_rate_hz,
            "decomposition": "LP=movmean(full,100); HP=full-LP",
            "padding_beats_each_side": PADDING_BEATS,
            "retained_beats_per_anatomy": RETAINED_BEATS,
            "samples_per_beat": FRAMES_PER_BEAT,
        },
        "counts": {
            "virtual_anatomies": sum(len(spec.anatomy_ids) for spec in splits),
            "retained_beats": sum(len(spec.anatomy_ids) for spec in splits)
            * RETAINED_BEATS,
            "retained_samples": sum(len(spec.anatomy_ids) for spec in splits)
            * RETAINED_BEATS
            * FRAMES_PER_BEAT,
            "splits": split_reports,
        },
        "split_anatomy_ids": {
            spec.name: spec.anatomy_ids.tolist() for spec in splits
        },
        "split_modes": {spec.name: spec.mode for spec in splits},
        "augmentation": augmentation,
        "noise": noise,
        "slow_nonvascular_amplitude_s_m": args.slow_nonvascular_amplitude,
        "slow_vascular_fraction": args.slow_vascular_fraction,
        "linearization_reference": args.linearization_reference,
        "jacobian_bank_size": len(bank),
        "linearization_cost": (
            "one exact resting forward/Jacobian per virtual anatomy"
            if args.linearization_reference == "anatomy"
            else "nearest entry from cached anatomical Jacobian bank"
        ),
        "linearized_acceleration_gate": {
            "maximum_component_nrmse": args.maximum_linearized_component_nrmse,
            "minimum_component_correlation": args.minimum_linearized_component_correlation,
            "comparisons": [
                {"split": split, "component": component, **comparison}
                for split, component, comparison in exact_comparisons
            ],
            "verified": bool(exact_comparisons),
        },
        "anti_inverse_crime": "fine forward mesh / coarse inverse mesh",
        "training_inverse_baseline": "homogeneous 0.7 S/m",
        "finger_config": str(finger_config.resolve()),
        "finger_config_sha256": _sha256(finger_config),
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "mesh_inverse_sha256": _sha256(Path(cfg.mesh_inv_h5)),
        "mesh_forward_sha256": _sha256(Path(cfg.mesh_fwd_h5)),
        "mapping_40_sha256": _sha256(Path(cfg.mappings_h5)),
        "seed": args.seed,
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    incomplete.unlink()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
