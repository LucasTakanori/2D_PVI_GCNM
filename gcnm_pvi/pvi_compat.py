"""Compatibility shims before importing pvi_forward (numpy 2 removed np.math)."""

import math

import numpy as np

if not hasattr(np, "math"):
    np.math = math  # noqa: A001
