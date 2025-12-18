#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan Core Model g-Screening (v0.4.2 Fix)

High-acceleration suppression for Solar-System extrapolation diagnostics.

Definition:
    S_g(g) = [1 + (g/g_c)^m]^{-1}
    a0_eff = a0_in * S_g(g)

Notes:
- This module does NOT implement Phi-screening.
- Use it as an optional safeguard in high-acceleration environments.
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def get_g_screening_factor(g_val: ArrayLike, g_c: float, m: float = 1.0) -> ArrayLike:
    """
    Compute screening factor:

        S_g(g) = [1 + (|g|/g_c)^m]^{-1}

    Behavior
    --------
    - Preserves shape for array inputs.
    - Returns NaN only for invalid *elements* (not whole-array NaN collapse).
    - Raises ValueError for invalid (g_c, m).

    Parameters
    ----------
    g_val : float or np.ndarray
        Newtonian acceleration g_N (m/s^2) (scalar or array).
    g_c : float
        Characteristic acceleration scale (m/s^2), must be > 0.
    m : float
        Screening index, must be > 0.

    Returns
    -------
    float or np.ndarray
        Screening factor S_g in (0,1], same shape as input.
    """
    if g_c is None or float(g_c) <= 0.0:
        raise ValueError(f"g_c must be > 0, got {g_c}")
    if m is None or float(m) <= 0.0:
        raise ValueError(f"g-screening index m must be > 0, got {m}")

    # Convert to array for elementwise safety; preserve scalar on return
    g_arr = np.asarray(g_val, dtype=float)
    out = np.full_like(g_arr, np.nan, dtype=float)

    good = np.isfinite(g_arr)
    if np.any(good):
        ratio = np.abs(g_arr[good]) / float(g_c)
        out[good] = 1.0 / (1.0 + ratio ** float(m))

    # Return scalar if scalar input
    if np.isscalar(g_val):
        return float(out.item())
    return out


def apply_g_screening(a0_in: ArrayLike, g_val: ArrayLike, g_c: float, m: float = 1.0) -> ArrayLike:
    """
    Apply to an input acceleration scale:

        a0_out = a0_in * S_g(g_val)

    Behavior
    --------
    - Supports scalar or array inputs for both a0_in and g_val.
    - Broadcasts as numpy would (e.g., scalar a0_in with array g_val).
    - Invalid elements propagate to NaN elementwise.

    Parameters
    ----------
    a0_in : float or np.ndarray
        Input acceleration scale a0 (m/s^2).
    g_val : float or np.ndarray
        Newtonian acceleration g_N (m/s^2).
    g_c : float
        High-g screening scale (m/s^2), must be > 0.
    m : float
        Screening index, must be > 0.

    Returns
    -------
    float or np.ndarray
        Screened acceleration scale a0_out (m/s^2).
    """
    a0_arr = np.asarray(a0_in, dtype=float)
    out = np.full(np.broadcast(a0_arr, np.asarray(g_val, dtype=float)).shape, np.nan, dtype=float)

    # Broadcast inputs
    a0_b = np.broadcast_to(a0_arr, out.shape)
    g_b = np.broadcast_to(np.asarray(g_val, dtype=float), out.shape)

    good = np.isfinite(a0_b) & np.isfinite(g_b) & (a0_b >= 0.0)
    if np.any(good):
        s_g = get_g_screening_factor(g_b[good], g_c=g_c, m=m)
        out[good] = a0_b[good] * np.asarray(s_g, dtype=float)

    if np.isscalar(a0_in) and np.isscalar(g_val):
        return float(out.item())
    return out
