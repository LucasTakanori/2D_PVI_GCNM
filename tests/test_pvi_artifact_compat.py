import csv
import json

from gcnm_pvi.train_pvi_bp import _verify_native_pvi_artifacts


CHECKPOINT_KEYS = {
    "dataset",
    "datetime",
    "epoch",
    "loss_func",
    "model",
    "name",
    "optimizer",
    "scheduler",
    "stopper",
    "tracker",
}


def test_current_pvi_artifact_contract(tmp_path):
    subject = "subject006"
    for directory in (
        "checkpoints",
        "configs",
        "history",
        "results",
        "statistics",
    ):
        (tmp_path / directory).mkdir()
    for suffix in ("", "_best"):
        (tmp_path / "checkpoints" / f"{subject}_checkpoints{suffix}.pth").touch()
    config = {
        key: {}
        for key in (
            "dataset",
            "datetime",
            "environment",
            "loss_func",
            "model",
            "optimizer",
            "scheduler",
            "stopper",
            "summary",
        )
    }
    (tmp_path / "configs" / f"{subject}_configs.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    with (tmp_path / "history" / f"{subject}_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        csv.writer(handle).writerow(
            [
                "epoch",
                "train_loss",
                "test_loss",
                "train_accuracy",
                "test_accuracy",
                "lr",
            ]
        )
    with (tmp_path / "results" / f"{subject}_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        csv.writer(handle).writerow(
            ["pred_1", "pred_2", "target_1", "target_2"]
        )
    statistics = {
        "num_train": 90,
        "num_test": 10,
        "num_periods": 100,
        "num_seq05": 100,
        "dbp_mae": 1.0,
        "sbp_mae": 2.0,
    }
    (tmp_path / "statistics" / f"{subject}_statistics.json").write_text(
        json.dumps(statistics), encoding="utf-8"
    )

    result = _verify_native_pvi_artifacts(
        tmp_path,
        subject,
        output_size=2,
        checkpoint={key: None for key in CHECKPOINT_KEYS},
    )
    assert result["status"] == "pass"
    assert result["schema"] == "current-pvi-ml-workflow-v3"
    assert result["result_columns"] == 4
    assert result["best_checkpoint_present"] is True

    # Upstream creates this file only after crossing the early-stopping
    # accuracy threshold. Max-epoch training still has a valid final checkpoint.
    (tmp_path / "checkpoints" / f"{subject}_checkpoints_best.pth").unlink()
    result = _verify_native_pvi_artifacts(
        tmp_path,
        subject,
        output_size=2,
        checkpoint={key: None for key in CHECKPOINT_KEYS},
    )
    assert result["status"] == "pass"
    assert result["best_checkpoint_present"] is False
