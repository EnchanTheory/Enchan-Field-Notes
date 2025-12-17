#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan Solar System Delta-Acceleration Diagnostic (v0.4.0)

Purpose
-------
Quantify the *anomalous* (non-Newtonian) sunward acceleration predicted by the
Enchan-style mapping (MOND-like closure used in v0.3.x / v0.4.x):

    g_enchan = sqrt(g_N^2 + a0_eff(r) * g_N)

This script implements the v0.4.0 split:

- Eq. 6.6.2 (galaxy-calibrated): Phi-screening / "pinning"
      a0_phi(r) = a0_free * S_phi(|Phi|)

- Eq. 6.6.3 (Solar-System extrapolation safeguard, optional): high-acceleration screening
      a0_eff(r) = a0_phi(r) * S_g(g_N)

By default, g-screening is OFF (g_c=None). Enable it explicitly via --g_c.

Usage:
  g-screening ON
  python enchan_solar_system_delta_accel.py --g_c 6e-11 --m 1.0

  g-screening OFF
  python enchan_solar_system_delta_accel.py

Outputs
-------
- CSV with radius, Newtonian acceleration, boost, delta_g, phantom mass proxy, and screening factors
- Diagnostic plots (boost, delta_g) if matplotlib is available

Notes
-----
- This is a diagnostic, not a full ephemeris fit.
- "phantom mass" is the point-mass equivalent that would reproduce delta_g at radius r:
      M_ph(r) = delta_g(r) * r^2 / G
"""

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
AU_M = 1.495978707e11
G_SI = 6.67430e-11
M_EARTH = 5.9722e24

# Solar gravitational parameter (m^3/s^2)
GM_SUN = 1.32712440018e20

# ---------------------------------------------------------------------
# v0.4.0 core imports (6.6.2 / 6.6.3 split)
# ---------------------------------------------------------------------
from enchan_core_model_plus import apply_screening as apply_phi_screening  # Eq. 6.6.2
from enchan_core_model_g_screening import apply_g_screening               # Eq. 6.6.3


@dataclass(frozen=True)
class BodyPoint:
    name: str
    r_au: float


DEFAULT_POINTS: List[BodyPoint] = [
    BodyPoint("Mercury", 0.387),
    BodyPoint("Earth", 1.0),
    BodyPoint("Jupiter", 5.204),
    BodyPoint("Pluto", 39.48),
    BodyPoint("Sedna", 506.0),
    BodyPoint("Planet 9?", 700.0),
]


def enchan_boost(g_newton: np.ndarray, a0_eff: np.ndarray) -> np.ndarray:
    """boost = g_enchan / g_N = sqrt(1 + a0_eff/g_N)."""
    g_newton = np.maximum(g_newton, 1e-30)
    return np.sqrt(1.0 + (a0_eff / g_newton))


def compute_profile(
    r_au: np.ndarray,
    a0_free: float,
    phi_c: float,
    n: float,
    g_c: Optional[float],
    m: float,
) -> pd.DataFrame:
    r_m = r_au * AU_M

    # Newtonian acceleration (m/s^2) and circular speed squared (km/s)^2
    g_N = GM_SUN / np.maximum(r_m, 1e-30) ** 2
    v_circ_sq_kms2 = (GM_SUN / np.maximum(r_m, 1e-30)) / 1.0e6  # (km/s)^2

    # Eq. 6.6.2: Phi-screening (pinning)
    a0_phi = apply_phi_screening(a0_free, phi_val=v_circ_sq_kms2, phi_c=phi_c, n=n)
    S_phi = np.asarray(a0_phi, dtype=float) / float(a0_free)

    # Eq. 6.6.3: optional high-acceleration screening
    if g_c is not None and g_c > 0:
        a0_eff = apply_g_screening(a0_phi, g_val=g_N, g_c=g_c, m=m)
        S_g = np.asarray(a0_eff, dtype=float) / np.maximum(np.asarray(a0_phi, dtype=float), 1e-300)
    else:
        a0_eff = a0_phi
        S_g = np.ones_like(g_N, dtype=float)

    S_total = np.asarray(a0_eff, dtype=float) / float(a0_free)

    boost = enchan_boost(g_N, a0_eff)
    g_en = boost * g_N
    delta_g = g_en - g_N  # m/s^2

    # point-mass equivalent ("phantom") in Earth masses
    M_ph_kg = delta_g * (r_m ** 2) / G_SI
    M_ph_Me = M_ph_kg / M_EARTH

    return pd.DataFrame({
        "r_AU": r_au,
        "r_m": r_m,
        "g_newton": g_N,
        "v_circ_sq_kms2": v_circ_sq_kms2,
        "S_phi": S_phi,
        "S_g": S_g,
        "S_total": S_total,
        "a0_free_ms2": a0_free,
        "a0_phi_ms2": a0_phi,
        "a0_eff_ms2": a0_eff,
        "boost": boost,
        "g_enchan": g_en,
        "delta_g": delta_g,
        "phantom_Me": M_ph_Me,
    })


def sample_points(df: pd.DataFrame, points: List[BodyPoint]) -> pd.DataFrame:
    """Nearest-neighbor sample at the named radii."""
    r = df["r_AU"].values
    rows = []
    for p in points:
        j = int(np.argmin(np.abs(r - p.r_au)))
        rows.append({
            "Body": p.name,
            "r(AU)": float(df.loc[j, "r_AU"]),
            "boost": float(df.loc[j, "boost"]),
            "Δg (m/s²)": float(df.loc[j, "delta_g"]),
            "phantom (M⊕)": float(df.loc[j, "phantom_Me"]),
            "S_phi": float(df.loc[j, "S_phi"]),
            "S_g": float(df.loc[j, "S_g"]),
            "S_total": float(df.loc[j, "S_total"]),
        })
    return pd.DataFrame(rows)


def plot_boost(df: pd.DataFrame, pts: List[BodyPoint], out_png: str) -> None:
    if not HAS_PLT:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(df["r_AU"].values, df["boost"].values)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("r (AU)")
    plt.ylabel("boost = g_enchan / g_N")
    plt.title("Enchan boost profile (diagnostic)")
    for p in pts:
        plt.axvline(p.r_au, linestyle="--", linewidth=1.0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_delta_g(df: pd.DataFrame, pts: List[BodyPoint], out_png: str) -> None:
    if not HAS_PLT:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(df["r_AU"].values, np.abs(df["delta_g"].values))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("r (AU)")
    plt.ylabel("|Δg| (m/s²)")
    plt.title("Enchan |Δg| profile (diagnostic)")
    for p in pts:
        plt.axvline(p.r_au, linestyle="--", linewidth=1.0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Enchan Solar System delta-acceleration profile (v0.4.0)")
    ap.add_argument("--a0", type=float, default=1.2e-10, help="Baseline a0_free (m/s^2)")
    ap.add_argument("--phi_c", type=float, default=40000.0, help="Phi-screening threshold Phi_c in (km/s)^2 (Eq. 6.6.2)")
    ap.add_argument("--n", type=float, default=1.0, help="Phi-screening index n>0 (Eq. 6.6.2)")
    ap.add_argument("--g_c", type=float, default=None, help="High-g screening scale g_c in (m/s^2) (Eq. 6.6.3). Omit to disable.")
    ap.add_argument("--m", type=float, default=1.0, help="High-g screening index m>0 (Eq. 6.6.3)")
    ap.add_argument("--rmin", type=float, default=0.1, help="Min radius (AU)")
    ap.add_argument("--rmax", type=float, default=1.0e4, help="Max radius (AU)")
    ap.add_argument("--npts", type=int, default=2000, help="Number of sample points (log-spaced)")
    ap.add_argument("--out", type=str, default="enchan_solar_system_delta_accel", help="Output prefix")
    ap.add_argument("--no_plot", action="store_true", help="Disable plots even if matplotlib is available")
    args = ap.parse_args()

    if args.n <= 0:
        raise SystemExit("ERROR: --n must be > 0")
    if args.g_c is not None and args.g_c <= 0:
        raise SystemExit("ERROR: --g_c must be > 0 if provided")
    if args.m <= 0:
        raise SystemExit("ERROR: --m must be > 0")

    r_au = np.logspace(math.log10(args.rmin), math.log10(args.rmax), int(args.npts))

    df = compute_profile(
        r_au=r_au,
        a0_free=args.a0,
        phi_c=args.phi_c,
        n=args.n,
        g_c=args.g_c,
        m=args.m,
    )

    out_csv = f"{args.out}.csv"
    out_boost_png = f"{args.out}_boost.png"
    out_dg_png = f"{args.out}_delta_g.png"

    df.to_csv(out_csv, index=False)

    print("\n--- Enchan Solar System Delta-Acceleration (v0.4.0) ---")
    if args.g_c is None:
        print(f"Params: a0={args.a0:.3e} m/s^2 | Phi_c={args.phi_c:.1f} (km/s)^2 | n={args.n:.2f} | g-screening=OFF")
    else:
        print(f"Params: a0={args.a0:.3e} m/s^2 | Phi_c={args.phi_c:.1f} (km/s)^2 | n={args.n:.2f} | g_c={args.g_c:.3e} m/s^2 | m={args.m:.2f}")

    tab = sample_points(df, DEFAULT_POINTS)
    # match user's preferred columns
    print(tab[["Body", "r(AU)", "boost", "Δg (m/s²)", "phantom (M⊕)", "S_phi", "S_g", "S_total"]].to_string(index=False))

    if (not args.no_plot) and HAS_PLT:
        plot_boost(df, DEFAULT_POINTS, out_boost_png)
        plot_delta_g(df, DEFAULT_POINTS, out_dg_png)
        print("")
        print(f"Saved: {out_boost_png}")
        print(f"Saved: {out_dg_png}")
    elif not HAS_PLT and (not args.no_plot):
        print("NOTE: matplotlib not available; skipping plots.")
    else:
        print("Plots disabled (--no_plot).")

    print(f"\nSaved: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
