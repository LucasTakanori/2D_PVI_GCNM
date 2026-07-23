import numpy as np

from gcnm_pvi.validate_synthetic_cohort_coverage import coverage_report


def _audit_signal(scale):
    return {
        "rms": {"subject_median": [scale] * 32},
        "quantiles": {
            "q0.01": {"subject_median": [-2 * scale] * 32},
            "q0.99": {"subject_median": [2 * scale] * 32},
        },
    }


def test_blind_coverage_gate_passes_signed_same_order_ranges():
    rng = np.random.default_rng(5)
    synthetic = {name: rng.normal(0, 1e-4, (500, 32)) for name in ("hp", "lp", "d_lp", "dd_lp")}
    audit = {
        "schema": "pvi-cohort-voltage-audit-v2",
        "global_subject_weighted": {
            "hp_referenced_mask05": _audit_signal(1e-4),
            "lp_referenced_mask05": _audit_signal(1e-4),
            "d_lp_referenced_mask05": _audit_signal(1e-4),
            "dd_lp_referenced_mask05": _audit_signal(1e-4),
        },
    }
    report = coverage_report(
        synthetic,
        audit,
        minimum_scale_ratio=0.02,
        maximum_scale_ratio=50,
        minimum_negative_fraction=0.001,
        maximum_negative_fraction=0.999,
    )
    assert report["status"] == "pass"


def test_blind_coverage_gate_rejects_orders_of_magnitude_error():
    synthetic = {name: np.ones((100, 32)) * 1e-10 for name in ("hp", "lp", "d_lp", "dd_lp")}
    audit = {
        "schema": "pvi-cohort-voltage-audit-v2",
        "global_subject_weighted": {
            "hp_referenced_mask05": _audit_signal(1e-3),
            "lp_referenced_mask05": _audit_signal(1e-3),
            "d_lp_referenced_mask05": _audit_signal(1e-3),
            "dd_lp_referenced_mask05": _audit_signal(1e-3),
        },
    }
    report = coverage_report(
        synthetic,
        audit,
        minimum_scale_ratio=0.02,
        maximum_scale_ratio=50,
        minimum_negative_fraction=0.001,
        maximum_negative_fraction=0.999,
    )
    assert report["status"] == "fail"
    assert report["failures"]
