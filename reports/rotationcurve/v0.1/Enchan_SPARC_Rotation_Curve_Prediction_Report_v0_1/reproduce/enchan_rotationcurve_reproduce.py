#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enchan SPARC rotation-curve prediction reproducibility script

Goal
----
Reproduce the fixed-parameter rotation-curve prediction test from SPARC Rotmod_LTG files:

  g_obs(r)  = V_obs(r)^2 / r
  g_bar(r)  = (V_gas^2 + Y_disk V_disk^2 + Y_bul V_bul^2) / r
  g_pred    = g_bar / (1 - exp(-sqrt(g_bar/a0)))
  V_pred(r) = sqrt(g_pred(r) * r)

This is a reproducibility/validation utility:
- public input: Rotmod_LTG.zip (SPARC mass-model files)
- deterministic outputs: CSV + figures + (optional) TeX stub
- no SciPy dependency (NumPy/Pandas/Matplotlib only)

Usage (example)
---------------
  python enchan_rotationcurve_reproduce.py --zip Rotmod_LTG.zip

Outputs are written to:
  Enchan_SPARC_Rotation_Curve_Prediction_Report_v0_1/

Notes
-----
- Rotmod_LTG.zip should NOT be committed (large upstream data).
- This script prints and stores the input sha256 for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


KPC_TO_M = 3.085677581491367e19
KMS_TO_MS = 1000.0


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_rotmod_dat(text: str) -> pd.DataFrame:
    """Parse a single *_rotmod.dat file from Rotmod_LTG.

    Expected columns (SPARC Rotmod_LTG):
      Rad, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul
    """
    rows = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = re.split(r"\s+", ln)
        if len(parts) < 6:
            continue
        # optional SB columns
        while len(parts) < 8:
            parts.append("nan")
        try:
            r_kpc = float(parts[0])
            vobs = float(parts[1])
            ev = float(parts[2])
            vgas = float(parts[3])
            vdisk = float(parts[4])
            vbul = float(parts[5])
            sbdisk = float(parts[6])
            sbbul = float(parts[7])
        except ValueError:
            continue
        rows.append((r_kpc, vobs, ev, vgas, vdisk, vbul, sbdisk, sbbul))
    return pd.DataFrame(
        rows,
        columns=["r_kpc", "Vobs_kms", "eVobs_kms", "Vgas_kms", "Vdisk_kms", "Vbul_kms", "SBdisk", "SBbul"],
    )


def load_rotmod_zip(zip_path: Path) -> pd.DataFrame:
    """Load all galaxies from Rotmod_LTG.zip into one DataFrame."""
    records: List[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if n.lower().endswith("_rotmod.dat")]
        for name in sorted(names):
            gal = Path(name).name.replace("_rotmod.dat", "")
            raw = z.read(name).decode("utf-8", errors="ignore")
            df = parse_rotmod_dat(raw)
            if df.empty:
                continue
            df.insert(0, "galaxy", gal)
            records.append(df)
    if not records:
        raise RuntimeError("No usable *_rotmod.dat files found inside the ZIP.")
    return pd.concat(records, ignore_index=True)


def g_from_vkpc(v_kms: np.ndarray, r_kpc: np.ndarray) -> np.ndarray:
    """Compute acceleration (m/s^2) from V^2/r with V in km/s and r in kpc."""
    v_ms = np.asarray(v_kms, dtype=float) * KMS_TO_MS
    r_m = np.asarray(r_kpc, dtype=float) * KPC_TO_M
    with np.errstate(divide="ignore", invalid="ignore"):
        return (v_ms ** 2) / r_m


def g_pred_from_gbar(gbar: np.ndarray, a0: float) -> np.ndarray:
    """Compute the empirical mapping used in the RAR literature."""
    gbar = np.asarray(gbar, dtype=float)
    out = np.full_like(gbar, np.nan, dtype=float)
    ok = np.isfinite(gbar) & (gbar > 0) & np.isfinite(a0) & (a0 > 0)
    x = np.sqrt(gbar[ok] / a0)
    denom = -np.expm1(-x)  # 1 - exp(-x), stable
    out[ok] = gbar[ok] / denom
    return out


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x ** 2)))


def make_tex(outdir: Path, stats: Dict[str, float], zip_name: str, sha: str, params: Dict[str, float]) -> None:
    """Write a minimal TeX stub (Field Notes-style preamble) for the run."""
    def _tex_escape(s: str) -> str:
        # Minimal escaping for filenames inside \texttt{...}
        return s.replace("_", r"\_")

    points_csv = f"sparc_vpred_points_Yd{params['Yd']:.2f}_Yb{params['Yb']:.2f}_a0fixed.csv"
    galaxy_csv = f"sparc_vpred_galaxy_summary_Yd{params['Yd']:.2f}_Yb{params['Yb']:.2f}_a0fixed.csv"

    tex_tpl = """%==============================================================================
% Enchan SPARC Rotation-Curve Prediction Test Report
% Copyright (c) 2025 Mitsuhiro Kobayashi
%
% This work (textual content, PDF) is licensed under a
% Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
% To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
%
% The LaTeX source code structure itself is available under the MIT License.
%==============================================================================

<BS>documentclass[11pt,a4paper]{article}

%===========================
% Packages (match Enchan Field Notes style)
%===========================
<BS>usepackage[utf8]{inputenc}
<BS>usepackage[T1]{fontenc}
<BS>usepackage{mathptmx}
<BS>usepackage{geometry}
<BS>geometry{margin=1in}
<BS>usepackage{amsmath,amssymb}
<BS>usepackage{bm}
<BS>usepackage{setspace}
<BS>usepackage{hyperref}
<BS>usepackage{booktabs}
<BS>usepackage{graphicx}

<BS>hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}

%===========================
% Helpers
%===========================
<BS>newcommand{<BS>chap}[1]{<BS>clearpage<BS>section{#1}}

%===========================
% Title / Author
%===========================
<BS>title{<BS>textbf{Enchan SPARC Rotation-Curve Prediction Report v0.1}<BS><BS>[0.5em]
<BS>large Fixed-parameter prediction of $V_{<BS>rm obs}(r)$ from public SPARC mass models}
<BS>author{Mitsuhiro Kobayashi<BS><BS>[0.25em]
Tokyo, Japan<BS><BS>
<BS>texttt{enchan.theory@gmail.com}}
<BS>date{<BS>today}

%===========================
% Document
%===========================
<BS>begin{document}

<BS>maketitle
<BS>thispagestyle{empty}

<BS>begin{abstract}
<BS>noindent
This report documents a minimal, reproducible ``rotation-curve prediction'' test using
public SPARC Newtonian mass-model files (<BS>texttt{Rotmod<BS>_LTG}).
For each radial point we compute the baryonic acceleration proxy
$g_{<BS>rm bar}(r)=<BS>left(V_{<BS>rm gas}^2+<BS>Upsilon_{<BS>rm disk}V_{<BS>rm disk}^2+<BS>Upsilon_{<BS>rm bul}V_{<BS>rm bul}^2<BS>right)/r$
and map it to a predicted gravitational response using the one-parameter empirical curve
$g_{<BS>rm pred}=g_{<BS>rm bar}/<BS>left(1-e^{-<BS>sqrt{g_{<BS>rm bar}/a_0}}<BS>right)$.
Unlike a fit, we hold parameters fixed at $(<BS>Upsilon_{<BS>rm disk},<BS>Upsilon_{<BS>rm bul})=(__Yd__,__Yb__)$ and
$a_0=__A0__~<BS>mathrm{m/s^2}$,
and compare the implied $V_{<BS>rm pred}(r)=<BS>sqrt{g_{<BS>rm pred}(r)r}$ against observed $V_{<BS>rm obs}(r)$.
Across __POINTS__ radial points from __GALAXIES__ galaxies, the global RMS residual in $<BS>log_{10} g$ is
__RMS_LOGG__ dex (velocity fractional RMS __RMS_FRACV__).
<BS>end{abstract}

<BS>clearpage
<BS>tableofcontents

<BS>chap{Scope and deliverable}
This document is intentionally narrow: it records a single verification task that can be rerun from public data.
The goal is to provide an external handle on an Enchan-relevant question:
can a direct baryons-to-dynamics mapping predict full rotation curves with fixed global parameters?

<BS>chap{Data and definitions}
We use the SPARC Rotmod files (one per galaxy) and compute $g_{<BS>rm obs}(r)=V_{<BS>rm obs}(r)^2/r$ and
$g_{<BS>rm bar}(r)=(V_{<BS>rm gas}^2+<BS>Upsilon_{<BS>rm disk}V_{<BS>rm disk}^2+<BS>Upsilon_{<BS>rm bul}V_{<BS>rm bul}^2)/r$
with unit conversion to <BS>mathrm{m/s^2}.

<BS>chap{Results}
<BS>begin{figure}[htbp]
<BS>centering
<BS>includegraphics[width=0.82<BS>linewidth]{fig_vpred_vs_vobs.png}
<BS>caption{Point-level comparison of predicted vs observed rotation speed.}
<BS>label{fig:vv}
<BS>end{figure}

<BS>begin{figure}[htbp]
<BS>centering
<BS>includegraphics[width=0.90<BS>linewidth]{fig_resid_logg_hist.png}
<BS>caption{Residual distribution in $<BS>log_{10} g$ (global RMS shown in text).}
<BS>label{fig:hist}
<BS>end{figure}

<BS>begin{figure}[htbp]
<BS>centering
<BS>includegraphics[width=0.90<BS>linewidth]{fig_galaxy_rms_hist.png}
<BS>caption{Distribution of per-galaxy RMS residuals in $<BS>log_{10} g$.}
<BS>label{fig:galhist}
<BS>end{figure}

<BS>chap{Reproducibility artifacts}
<BS>begin{itemize}
<BS>item <BS>texttt{__POINTS_CSV__}
<BS>item <BS>texttt{__GALAXY_CSV__}
<BS>end{itemize}

<BS>chap{References}
<BS>begin{itemize}
<BS>item SPARC database (downloads): <BS>url{https://astroweb.case.edu/SPARC/}
<BS>end{itemize}

<BS>end{document}
"""

    tex = tex_tpl.replace("<BS>", "\\")
    tex = tex.replace("__Yd__", f"{params['Yd']:.2f}")
    tex = tex.replace("__Yb__", f"{params['Yb']:.2f}")
    tex = tex.replace("__A0__", f"{params['a0']:.3e}")
    tex = tex.replace("__POINTS__", str(int(stats["points"])))
    tex = tex.replace("__GALAXIES__", str(int(stats["galaxies"])))
    tex = tex.replace("__RMS_LOGG__", f"{stats['rms_logg']:.3f}")
    tex = tex.replace("__RMS_FRACV__", f"{stats['rms_fracV']:.3f}")
    tex = tex.replace("__POINTS_CSV__", _tex_escape(points_csv))
    tex = tex.replace("__GALAXY_CSV__", _tex_escape(galaxy_csv))

    (outdir / "enchan_sparc_rotationcurve_prediction_report_v0p1.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to Rotmod_LTG.zip")
    ap.add_argument("--outdir", default="Enchan_SPARC_Rotation_Curve_Prediction_Report_v0_1",
                    help="Output directory")
    ap.add_argument("--Yd", type=float, default=0.60, help="Disk mass-to-light ratio (multiplies Vdisk^2)")
    ap.add_argument("--Yb", type=float, default=0.70, help="Bulge mass-to-light ratio (multiplies Vbul^2)")
    ap.add_argument("--a0", type=float, default=1.12e-10, help="Fixed a0 (m/s^2)")
    ap.add_argument("--make-tex", action="store_true", help="Also write a minimal TeX report stub")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing input ZIP: {zip_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sha = sha256_file(zip_path)

    df = load_rotmod_zip(zip_path)

    # Basic cuts
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["r_kpc", "Vobs_kms", "eVobs_kms", "Vgas_kms", "Vdisk_kms", "Vbul_kms"]).copy()
    df = df[(df["r_kpc"] > 0) & (df["Vobs_kms"] > 0)].copy()

    # Accelerations
    gobs = g_from_vkpc(df["Vobs_kms"].to_numpy(float), df["r_kpc"].to_numpy(float))

    vgas2 = df["Vgas_kms"].to_numpy(float) ** 2
    vdisk2 = df["Vdisk_kms"].to_numpy(float) ** 2
    vbul2 = df["Vbul_kms"].to_numpy(float) ** 2
    vbary2 = np.maximum(vgas2 + args.Yd * vdisk2 + args.Yb * vbul2, 0.0)
    vbary = np.sqrt(vbary2)
    gbar = g_from_vkpc(vbary, df["r_kpc"].to_numpy(float))

    gpred = g_pred_from_gbar(gbar, args.a0)

    # Vpred
    r_m = df["r_kpc"].to_numpy(float) * KPC_TO_M
    with np.errstate(invalid="ignore"):
        Vpred_ms = np.sqrt(gpred * r_m)
    Vpred_kms = Vpred_ms / KMS_TO_MS

    df["gobs"] = gobs
    df["gbar"] = gbar
    df["gpred"] = gpred
    df["Vpred_kms"] = Vpred_kms

    # Residuals
    with np.errstate(divide="ignore", invalid="ignore"):
        df["resid_logg"] = np.log10(df["gobs"]) - np.log10(df["gpred"])
        df["resid_fracV"] = (df["Vobs_kms"] - df["Vpred_kms"]) / df["Vobs_kms"]

    rms_logg = rms(df["resid_logg"].to_numpy(float))
    rms_fracV = rms(df["resid_fracV"].to_numpy(float))

    # Per-galaxy metrics
    ln10 = math.log(10.0)
    df["sigma_logg"] = (2.0 * df["eVobs_kms"] / df["Vobs_kms"]) / ln10

    gstats = []
    for gal, g in df.groupby("galaxy"):
        resid = g["resid_logg"].to_numpy(float)
        sig = g["sigma_logg"].to_numpy(float)
        gal_rms = rms(resid)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = resid / sig
        z = z[np.isfinite(z)]
        chi2_red = float(np.mean(z ** 2)) if z.size > 0 else float("nan")
        gstats.append((gal, int(len(g)), gal_rms, chi2_red))

    gdf = pd.DataFrame(gstats, columns=["galaxy", "n_points", "rms_logg_dex", "chi2_red_approx"])

    frac_chi2_lt_2 = float(np.mean(gdf["chi2_red_approx"] < 2.0))
    frac_chi2_lt_5 = float(np.mean(gdf["chi2_red_approx"] < 5.0))
    med_chi2 = float(np.nanmedian(gdf["chi2_red_approx"].to_numpy(float)))

    # Write outputs
    tag = f"Yd{args.Yd:.2f}_Yb{args.Yb:.2f}_a0fixed"
    points_csv = outdir / f"sparc_vpred_points_{tag}.csv"
    gal_csv = outdir / f"sparc_vpred_galaxy_summary_{tag}.csv"
    global_csv = outdir / f"sparc_vpred_global_summary_{tag}.csv"

    df.to_csv(points_csv, index=False)
    gdf.to_csv(gal_csv, index=False)

    global_summary = pd.DataFrame([{
        "zip": zip_path.name,
        "sha256": sha,
        "Yd": args.Yd,
        "Yb": args.Yb,
        "a0_m_per_s2": args.a0,
        "galaxies": int(gdf.shape[0]),
        "points": int(df.shape[0]),
        "rms_logg_dex": rms_logg,
        "rms_fracV": rms_fracV,
        "median_chi2_red_approx": med_chi2,
        "frac_chi2_red_lt_2": frac_chi2_lt_2,
        "frac_chi2_red_lt_5": frac_chi2_lt_5,
    }])
    global_summary.to_csv(global_csv, index=False)

    # Figures (matplotlib default colors)
    plt.figure()
    plt.scatter(df["Vobs_kms"], df["Vpred_kms"], s=8, alpha=0.5)
    lo = float(np.nanmin(df[["Vobs_kms", "Vpred_kms"]].to_numpy()))
    hi = float(np.nanmax(df[["Vobs_kms", "Vpred_kms"]].to_numpy()))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Vobs [km/s]")
    plt.ylabel("Vpred [km/s]")
    plt.title("SPARC rotation-curve prediction (fixed parameters)")
    plt.tight_layout()
    plt.savefig(outdir / "fig_vpred_vs_vobs.png", dpi=200)
    plt.close()

    plt.figure()
    x = df["resid_logg"].to_numpy(float)
    x = x[np.isfinite(x)]
    plt.hist(x, bins=50)
    plt.xlabel("Residual in log10(g) [dex]")
    plt.ylabel("Count")
    plt.title(f"Residuals (global RMS = {rms_logg:.3f} dex)")
    plt.tight_layout()
    plt.savefig(outdir / "fig_resid_logg_hist.png", dpi=200)
    plt.close()

    plt.figure()
    y = gdf["rms_logg_dex"].to_numpy(float)
    y = y[np.isfinite(y)]
    plt.hist(y, bins=35)
    plt.xlabel("Per-galaxy RMS residual in log10(g) [dex]")
    plt.ylabel("Galaxies")
    plt.title("Per-galaxy RMS residual distribution")
    plt.tight_layout()
    plt.savefig(outdir / "fig_galaxy_rms_hist.png", dpi=200)
    plt.close()

    if args.make_tex:
        make_tex(
            outdir=outdir,
            stats={"galaxies": float(gdf.shape[0]), "points": float(df.shape[0]), "rms_logg": rms_logg, "rms_fracV": rms_fracV},
            zip_name=zip_path.name,
            sha=sha,
            params={"Yd": args.Yd, "Yb": args.Yb, "a0": args.a0},
        )

    print("")
    print("Rotation-curve prediction reproduce summary")
    print(f"  zip: {zip_path.name}")
    print(f"  sha256: {sha}")
    print(f"  galaxies: {int(gdf.shape[0])}")
    print(f"  points: {int(df.shape[0])}")
    print("")
    print(f"  fixed params: Yd={args.Yd:.2f}, Yb={args.Yb:.2f}, a0={args.a0:.3e} m/s^2")
    print(f"  global RMS resid log10(g): {rms_logg:.3f} dex")
    print(f"  global RMS frac V: {rms_fracV:.3f}")
    print(f"  median chi2_red_approx: {med_chi2:.2f}")
    print(f"  frac chi2_red_approx <2: {frac_chi2_lt_2:.3f}")
    print(f"  frac chi2_red_approx <5: {frac_chi2_lt_5:.3f}")
    print("")
    print(f"Outputs written to: {outdir.resolve()}")
    print("Key files:")
    print(f"  - {points_csv.name}")
    print(f"  - {gal_csv.name}")
    print(f"  - {global_csv.name}")
    print("  - fig_vpred_vs_vobs.png, fig_resid_logg_hist.png, fig_galaxy_rms_hist.png")
    if args.make_tex:
        print("  - enchan_sparc_rotationcurve_prediction_report_v0p1.tex")


if __name__ == "__main__":
    main()
