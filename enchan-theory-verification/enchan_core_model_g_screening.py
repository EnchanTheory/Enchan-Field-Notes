#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan Core Model g-Screening (v0.4.0 / Eq. 6.6.3 ONLY)

High-acceleration suppression for Solar-System extrapolation tests.

Eq. 6.6.3:
    S_g(g) = [1 + (g/g_c)^m]^{-1}
    a0_eff = a0_in * S_g(g)

Notes:
- This module does NOT implement Phi-screening (Eq. 6.6.2).
- Use it as an optional safeguard in high-acceleration environments.
"""

from __future__ import annotations
import numpy as np
from typing import Union

def get_g_screening_factor(
    g_val: Union[float, np.ndarray],
    g_c: float,
    m: float = 1.0
) -> Union[float, np.ndarray]:
    """
    Eq. 6.6.3:
        S_g(g) = [1 + (|g|/g_c)^m]^{-1}

    Returns NaN for invalid inputs.
    """
    if g_c is None or g_c <= 0:
        raise ValueError(f"g_c must be > 0, got {g_c}")
    if m is None or m <= 0:
        raise ValueError(f"g-screening index 'm' must be > 0, got {m}")

    if g_val is None:
        return float("nan")
    if not np.all(np.isfinite(g_val)):
        return float("nan")

    return 1.0 / (1.0 + (np.abs(g_val) / float(g_c)) ** float(m))


def apply_g_screening(
    a0_in: Union[float, np.ndarray],
    g_val: Union[float, np.ndarray],
    g_c: float,
    m: float = 1.0
) -> Union[float, np.ndarray]:
    """
    Apply Eq. 6.6.3 to an input acceleration scale:
        a0_out = a0_in * S_g(g)
    """
    if not np.all(np.isfinite(a0_in)):
        return float("nan")

    s_g = get_g_screening_factor(g_val=g_val, g_c=g_c, m=m)
    return a0_in * s_g
