#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan rotation-curve prediction reproduce (ENCHAN model)

Given SPARC Rotmod_LTG mass models, predict V(r) from baryons using the Enchan quadrature law:

  g_bar(r) = (Vgas^2 + Yd Vdisk^2 + Yb Vbul^2) / r
  g_pred(r) = sqrt( g_bar(r)^2 + a0*g_bar(r) )
  V_pred(r) = sqrt( g_pred(r) * r )

This version is the Enchan-derived alternative to the earlier "baseline RAR interpolation" mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from io import StringIO
from pathlib import Path

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
    ap.add_argument("--outdir", default="Enchan_SPARC_Rotation_Curve_Prediction_Report_v0_1", help="Output directory")
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

    r_m = df["r_kpc"].to_numpy(dtype=float) * KPC_TO_M
    Vobs = df["Vobs"].to_numpy(dtype=float) * KMS_TO_MS
    Vgas = df["Vgas"].to_numpy(dtype=float) * KMS_TO_MS
    Vdisk = df["Vdisk"].to_numpy(dtype=float) * KMS_TO_MS
    Vbul = df["Vbul"].to_numpy(dtype=float) * KMS_TO_MS

    g_obs = (Vobs ** 2) / r_m
    g_bar = (Vgas ** 2 + args.Yd * (Vdisk ** 2) + args.Yb * (Vbul ** 2)) / r_m
    g_pred = g_pred_quadrature(g_bar, args.a0)

    V_pred = np.sqrt(g_pred * r_m)

    ok = np.isfinite(Vobs) & np.isfinite(V_pred) & (Vobs > 0) & (V_pred > 0) & np.isfinite(g_obs) & np.isfinite(g_pred) & (g_obs > 0) & (g_pred > 0)
    df2 = df.loc[ok].copy().reset_index(drop=True)

    df2["Vobs_ms"] = Vobs[ok]
    df2["Vpred_ms"] = V_pred[ok]
    df2["g_obs"] = g_obs[ok]
    df2["g_bar"] = g_bar[ok]
    df2["g_pred_enchan"] = g_pred[ok]

    df2["resid_logg"] = np.log10(df2["g_obs"]) - np.log10(df2["g_pred_enchan"])
    df2["fracV"] = (df2["Vobs_ms"] - df2["Vpred_ms"]) / df2["Vobs_ms"]

    rms_logg = float(np.sqrt(np.mean(df2["resid_logg"].to_numpy() ** 2)))
    rms_fracV = float(np.sqrt(np.mean(df2["fracV"].to_numpy() ** 2)))

    df2.to_csv(outdir / "sparc_vpred_points_enchan.csv", index=False)

    # V_pred vs V_obs
    plt.figure()
    plt.scatter(df2["Vobs_ms"] / KMS_TO_MS, df2["Vpred_ms"] / KMS_TO_MS, s=10, alpha=0.6)
    # y=x line
    vmin = float(np.nanmin(df2["Vobs_ms"] / KMS_TO_MS))
    vmax = float(np.nanmax(df2["Vobs_ms"] / KMS_TO_MS))
    xs = np.linspace(vmin, vmax, 200)
    plt.plot(xs, xs)
    plt.xlabel("Vobs [km/s]")
    plt.ylabel("Vpred [km/s]")
    plt.title("Rotation-curve prediction (Enchan quadrature law)")
    plt.tight_layout()
    plt.savefig(outdir / "fig_vpred_vs_vobs_enchan.png", dpi=200)
    plt.close()

    # Residual histograms
    plt.figure()
    plt.hist(df2["resid_logg"].to_numpy(), bins=30)
    plt.xlabel("Residual: log10(g_obs) - log10(g_pred)")
    plt.ylabel("Count")
    plt.title(f"Residuals (Enchan) RMS={rms_logg:.3f} dex")
    plt.tight_layout()
    plt.savefig(outdir / "fig_resid_logg_hist_enchan.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(df2["fracV"].to_numpy(), bins=30)
    plt.xlabel("Fractional V residual: (Vobs - Vpred)/Vobs")
    plt.ylabel("Count")
    plt.title(f"Fractional V residuals RMS={rms_fracV:.3f}")
    plt.tight_layout()
    plt.savefig(outdir / "fig_fracV_hist_enchan.png", dpi=200)
    plt.close()

    # Galaxy summary
    gsum = []
    for gal, g in df2.groupby("galaxy"):
        rr = g["resid_logg"].to_numpy()
        vv = g["fracV"].to_numpy()
        gsum.append({
            "galaxy": gal,
            "N": int(len(g)),
            "rms_logg_dex": float(np.sqrt(np.mean(rr**2))),
            "rms_fracV": float(np.sqrt(np.mean(vv**2))),
        })
    gsum_df = pd.DataFrame(gsum).sort_values("rms_logg_dex")
    gsum_df.to_csv(outdir / "sparc_vpred_galaxy_summary_enchan.csv", index=False)

    global_sum = pd.DataFrame([{
        "zip_file": zip_path.name,
        "zip_sha256": sha256_file(zip_path),
        "galaxies": int(df2["galaxy"].nunique()),
        "points": int(len(df2)),
        "Yd": float(args.Yd),
        "Yb": float(args.Yb),
        "a0_SI": float(args.a0),
        "model": "enchan_quadrature",
        "global_rms_logg_dex": rms_logg,
        "global_rms_fracV": rms_fracV,
    }])
    global_sum.to_csv(outdir / "sparc_vpred_global_summary_enchan.csv", index=False)

    print("Rotation-curve prediction reproduce summary (ENCHAN)")
    print(f"  zip: {zip_path.name}")
    print(f"  sha256: {sha256_file(zip_path)}")
    print(f"  galaxies: {int(df2['galaxy'].nunique())}")
    print(f"  points: {int(len(df2))}")
    print("")
    print(f"  model: g_pred = sqrt(g_bar^2 + a0*g_bar)")
    print(f"  params: Yd={args.Yd:.2f}, Yb={args.Yb:.2f}, a0={args.a0:.3e} m/s^2")
    print(f"  global RMS resid log10(g): {rms_logg:.3f} dex")
    print(f"  global RMS frac V: {rms_fracV:.3f}")
    print("")
    print(f"Outputs written to: {outdir.resolve()}")
    print("Key files:")
    print("  - sparc_vpred_points_enchan.csv")
    print("  - sparc_vpred_galaxy_summary_enchan.csv")
    print("  - sparc_vpred_global_summary_enchan.csv")
    print("  - fig_vpred_vs_vobs_enchan.png, fig_resid_logg_hist_enchan.png, fig_fracV_hist_enchan.png")


if __name__ == "__main__":
    main()
