"""Repository-wide pytest ordering and test-environment contracts."""

from __future__ import annotations


def pytest_collection_modifyitems(items) -> None:
    """Run fork-based coverage last to avoid inherited BLAS thread locks.

    The production process-executor path intentionally exercises POSIX fork.
    Forking after Torch/SciPy have initialized worker threads can poison locks
    inherited by the test process and make a later FEM solve wait forever.
    Keeping this marked test last preserves its coverage and lets the normal
    test command terminate deterministically.
    """

    items.sort(
        key=lambda item: item.get_closest_marker("fork_isolation") is not None
    )
