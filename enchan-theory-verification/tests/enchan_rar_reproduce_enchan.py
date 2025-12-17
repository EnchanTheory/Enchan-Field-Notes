#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan RAR reproduce (ENCHAN model)

This script evaluates an Enchan-derived acceleration closure on SPARC Rotmod_LTG:

- compute g_obs(r) = Vobs^2 / r
- compute g_bar(r) = (Vgas^2 + Yd Vdisk^2 + Yb Vbul^2) / r
- Enchan prediction (quadrature law):
      g_pred = sqrt(g_bar^2 + a0*g_bar)

Unlike the earlier "baseline" package, this is NOT the empirical RAR interpolation curve.
It is a geometry-first closure that uses a single constant a0.

Recommended workflow:
1) Run BTFR Enchan script to estimate a0 from BTFR:
     python enchan_btfr_reproduce_enchan.py --mrt BTFR_Lelli2019.mrt
   -> produces btfr_a0_summary.csv (median a0)

2) Run this script with that a0:
     python enchan_rar_reproduce_enchan.py --zip Rotmod_LTG.zip --a0 1.53e-10
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from io import StringIO
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from enchan_core_model import KMS_TO_MS, g_pred_quadrature


KPC_TO_M = 3.0856775814913673e19


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rotmod_zip(zip_path: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.endswith("_rotmod.dat"):
                continue
            gal = name.replace("_rotmod.dat", "")
            raw = z.read(name).decode("utf-8", errors="ignore")
            data_lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
            if not data_lines:
                continue
            df = pd.read_csv(
                StringIO("\n".join(data_lines)),
                sep=r"\s+",
                header=None,
                names=["r_kpc", "Vobs", "eV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"],
                engine="python",
            )
            df["galaxy"] = gal
            rows.append(df)
    if not rows:
        raise RuntimeError("No *_rotmod.dat files found in the ZIP.")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to Rotmod_LTG.zip (not committed)")
    ap.add_argument("--outdir", default="Enchan_RAR_Test_Report_v0_1", help="Output directory")
    ap.add_argument("--Yd", type=float, default=0.60, help="Disk mass-to-light ratio (multiplies Vdisk^2)")
    ap.add_argument("--Yb", type=float, default=0.70, help="Bulge mass-to-light ratio (multiplies Vbul^2)")
    ap.add_argument("--a0", type=float, required=True, help="Enchan a0 [m/s^2] (suggested: BTFR median)")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing input ZIP: {zip_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_rotmod_zip(zip_path)

    # Compute accelerations
    r_m = df["r_kpc"].to_numpy(dtype=float) * KPC_TO_M
    Vobs = df["Vobs"].to_numpy(dtype=float) * KMS_TO_MS
    Vgas = df["Vgas"].to_numpy(dtype=float) * KMS_TO_MS
    Vdisk = df["Vdisk"].to_numpy(dtype=float) * KMS_TO_MS
    Vbul = df["Vbul"].to_numpy(dtype=float) * KMS_TO_MS

    g_obs = (Vobs ** 2) / r_m
    g_bar = (Vgas ** 2 + args.Yd * (Vdisk ** 2) + args.Yb * (Vbul ** 2)) / r_m
    g_pred = g_pred_quadrature(g_bar, args.a0)

    # Clean
    ok = np.isfinite(g_obs) & np.isfinite(g_bar) & np.isfinite(g_pred) & (g_obs > 0) & (g_bar > 0) & (g_pred > 0)
    df2 = df.loc[ok].copy().reset_index(drop=True)
    df2["g_obs"] = g_obs[ok]
    df2["g_bar"] = g_bar[ok]
    df2["g_pred_enchan"] = g_pred[ok]
    df2["resid_logg"] = np.log10(df2["g_obs"]) - np.log10(df2["g_pred_enchan"])

    rms = float(np.sqrt(np.mean(df2["resid_logg"].to_numpy() ** 2)))

    # Save points
    df2.to_csv(outdir / "sparc_rar_points_processed_enchan.csv", index=False)

    # Plot: g_obs vs g_bar (RAR plane)
    plt.figure()
    plt.scatter(np.log10(df2["g_bar"]), np.log10(df2["g_obs"]), s=8, alpha=0.5)
    # Enchan curve line
    xs = np.linspace(np.nanmin(np.log10(df2["g_bar"])) - 0.2, np.nanmax(np.log10(df2["g_bar"])) + 0.2, 400)
    gb_line = 10 ** xs
    gp_line = g_pred_quadrature(gb_line, args.a0)
    plt.plot(xs, np.log10(gp_line))
    plt.xlabel("log10(g_bar [m/s^2])")
    plt.ylabel("log10(g_obs [m/s^2])")
    plt.title("SPARC RAR with Enchan quadrature law")
    plt.tight_layout()
    plt.savefig(outdir / "fig_rar_points_enchan.png", dpi=200)
    plt.close()

    # Residual histogram
    plt.figure()
    plt.hist(df2["resid_logg"].to_numpy(), bins=30)
    plt.xlabel("Residual: log10(g_obs) - log10(g_pred)")
    plt.ylabel("Count")
    plt.title(f"RAR residuals (Enchan) RMS={rms:.3f} dex")
    plt.tight_layout()
    plt.savefig(outdir / "fig_rar_resid_hist_enchan.png", dpi=200)
    plt.close()

    # Summary CSV
    summary = pd.DataFrame([{
        "zip_file": zip_path.name,
        "zip_sha256": sha256_file(zip_path),
        "galaxies": int(df2["galaxy"].nunique()),
        "points": int(len(df2)),
        "Yd": float(args.Yd),
        "Yb": float(args.Yb),
        "a0_SI": float(args.a0),
        "model": "enchan_quadrature",
        "rms_dex_logg": rms,
    }])
    summary.to_csv(outdir / "rar_enchan_summary.csv", index=False)

    print("RAR reproduce summary (ENCHAN)")
    print(f"  zip: {zip_path.name}")
    print(f"  sha256: {sha256_file(zip_path)}")
    print(f"  galaxies: {int(df2['galaxy'].nunique())}")
    print(f"  points: {int(len(df2))}")
    print("")
    print(f"  model: g_pred = sqrt(g_bar^2 + a0*g_bar)")
    print(f"  params: Yd={args.Yd:.2f}, Yb={args.Yb:.2f}, a0={args.a0:.3e} m/s^2")
    print(f"  RMS resid log10(g): {rms:.3f} dex")
    print(f"Outputs written to: {outdir.resolve()}")
    print("Key files:")
    print("  - sparc_rar_points_processed_enchan.csv")
    print("  - rar_enchan_summary.csv")
    print("  - fig_rar_points_enchan.png, fig_rar_resid_hist_enchan.png")


if __name__ == "__main__":
    main()
