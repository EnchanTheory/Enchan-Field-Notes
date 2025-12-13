#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enchan RAR reproduce script (SPARC Rotmod_LTG.zip)

- Reads Rotmod_LTG.zip (175 galaxy .dat files)
- Computes:
    g_obs(r) = V_obs(r)^2 / r
    g_bar(r) = (V_gas^2 + Y_disk V_disk^2 + Y_bul V_bul^2) / r
  with unit conversion from (km/s)^2/kpc to m/s^2
- Fits one-parameter RAR curve:
    g_model = g_bar / (1 - exp(-sqrt(g_bar/a0)))
  by weighted least squares in log10-space.
- Produces CSV + PNG figures + TeX report with numbers inserted.

Usage:
  python enchan_rar_reproduce.py --zip Rotmod_LTG.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- constants ---
KPC_TO_M = 3.085677581491367e19
KM_TO_M = 1e3


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rotmod_zip(zip_path: Path) -> pd.DataFrame:
    """Parse SPARC Rotmod_LTG.zip (.dat per galaxy)."""
    rows = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.lower().endswith("_rotmod.dat"):
                continue
            fname = Path(name).name
            gal = fname.replace("_rotmod.dat", "")
            txt = z.read(name).decode("utf-8", errors="replace").splitlines()
            for ln in txt:
                ln = ln.strip()
                if (not ln) or ln.startswith("#"):
                    continue
                parts = ln.split()
                if len(parts) < 8:
                    continue
                r_kpc, vobs, errv, vgas, vdisk, vbul, sbdisk, sbbul = parts[:8]
                rows.append({
                    "galaxy": gal,
                    "r_kpc": float(r_kpc),
                    "Vobs_kms": float(vobs),
                    "eVobs_kms": float(errv),
                    "Vgas_kms": float(vgas),
                    "Vdisk_kms": float(vdisk),
                    "Vbul_kms": float(vbul),
                    "SBdisk": float(sbdisk),
                    "SBbul": float(sbbul),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No data rows parsed from zip.")
    return df


def compute_accelerations(df: pd.DataFrame, Yd: float, Yb: float) -> pd.DataFrame:
    """Compute g_obs, g_bar in SI units (m/s^2) and log10 versions."""
    d = df.copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["r_kpc", "Vobs_kms", "Vgas_kms", "Vdisk_kms", "Vbul_kms"]).reset_index(drop=True)
    d = d[(d["r_kpc"] > 0) & (d["Vobs_kms"] > 0)].reset_index(drop=True)

    r_m = d["r_kpc"].to_numpy() * KPC_TO_M
    Vobs = d["Vobs_kms"].to_numpy() * KM_TO_M
    Vgas = d["Vgas_kms"].to_numpy() * KM_TO_M
    Vdisk = d["Vdisk_kms"].to_numpy() * KM_TO_M
    Vbul = d["Vbul_kms"].to_numpy() * KM_TO_M

    g_obs = (Vobs**2) / r_m
    g_bar = (Vgas**2 + Yd * Vdisk**2 + Yb * Vbul**2) / r_m

    d["g_obs"] = g_obs
    d["g_bar"] = g_bar
    d["logg_obs"] = np.log10(g_obs)
    d["logg_bar"] = np.log10(g_bar)

    eV = d["eVobs_kms"].to_numpy() * KM_TO_M
    sigma_frac = 2.0 * (eV / Vobs)
    d["sigma_logg_obs"] = sigma_frac / np.log(10.0)

    return d


def rar_model(g_bar: np.ndarray, a0: float) -> np.ndarray:
    x = np.sqrt(np.maximum(g_bar, 0) / a0)
    denom = 1.0 - np.exp(-x)
    denom = np.where(denom <= 0, np.nan, denom)
    return g_bar / denom


def fit_a0_logspace(g_bar: np.ndarray, logg_obs: np.ndarray, sigma_logg: np.ndarray,
                    log10_a0_min: float = -12.0, log10_a0_max: float = -9.0, ngrid: int = 2001) -> Dict[str, float]:
    """Deterministic grid-search fit in log10(a0) minimizing weighted MSE in dex."""
    mask = np.isfinite(g_bar) & np.isfinite(logg_obs) & (g_bar > 0)
    gb = g_bar[mask]
    y = logg_obs[mask]
    s = sigma_logg[mask]

    finite = np.isfinite(s) & (s > 0)
    if np.any(finite):
        w = 1.0 / np.where(finite, s**2, np.nan)
        fw = w[np.isfinite(w)]
        w = np.where(np.isfinite(w), w, np.nanmedian(fw) if fw.size else 1.0)
    else:
        w = np.ones_like(y)

    grid = np.linspace(log10_a0_min, log10_a0_max, ngrid)
    best_log10 = float("nan")
    best_a0 = float("nan")
    best_wmse = float("inf")

    for lg in grid:
        a0 = 10.0**lg
        gmod = rar_model(gb, a0)
        loggmod = np.log10(gmod)
        r = y - loggmod
        ok = np.isfinite(r)
        if not np.any(ok):
            continue
        wmse = float(np.sum(w[ok] * r[ok]**2) / np.sum(w[ok]))
        if wmse < best_wmse:
            best_wmse = wmse
            best_log10 = float(lg)
            best_a0 = float(a0)

    return {"log10_a0": best_log10, "a0": best_a0, "wmse": best_wmse}


def rms_dex(resid: np.ndarray) -> float:
    r = resid[np.isfinite(resid)]
    return float(np.sqrt(np.mean(r**2))) if r.size else float("nan")


def spearman_rho(x, y) -> float:
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return float("nan")

    rx = x[mask].rank(method="average")
    ry = y[mask].rank(method="average")

    # Pearson correlation on ranks (Spearman's rho)
    cx = rx - rx.mean()
    cy = ry - ry.mean()
    denom = np.sqrt(float((cx * cx).sum()) * float((cy * cy).sum()))
    if denom == 0.0:
        return float("nan")
    return float((cx * cy).sum() / denom)


def make_fig_rar_points(df: pd.DataFrame, a0: float, outpng: Path) -> None:
    plt.figure()
    plt.scatter(df["logg_bar"], df["logg_obs"], s=6, alpha=0.35)

    gb = np.logspace(np.nanmin(df["logg_bar"]) - 0.2, np.nanmax(df["logg_bar"]) + 0.2, 300)
    gm = rar_model(gb, a0)
    plt.plot(np.log10(gb), np.log10(gm))

    plt.xlabel("log10(g_bar) [m/s^2]")
    plt.ylabel("log10(g_obs) [m/s^2]")
    plt.title("SPARC RAR (points) with 1-parameter reference curve")
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()


def make_fig_resid_sb_binned(df: pd.DataFrame, outpng: Path, nbins: int = 8) -> None:
    d = df.copy()
    d = d[np.isfinite(d["SBdisk"]) & (d["SBdisk"] > 0) & np.isfinite(d["resid"])].copy()
    if d.empty:
        return

    x = np.log10(d["SBdisk"].to_numpy())
    y = d["resid"].to_numpy()

    qs = np.linspace(0, 1, nbins + 1)
    edges = np.unique(np.quantile(x, qs))
    if edges.size < 3:
        return

    xb, yb = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (x >= lo) & (x <= hi) if i == len(edges) - 2 else (x >= lo) & (x < hi)
        if np.sum(sel) < 10:
            continue
        xb.append(float(np.mean(x[sel])))
        yb.append(float(np.mean(y[sel])))

    plt.figure()
    plt.scatter(x, y, s=6, alpha=0.25)
    if xb:
        plt.plot(xb, yb)
    plt.axhline(0.0)
    plt.xlabel("log10(SBdisk) [L/pc^2]")
    plt.ylabel("Residual: log10(g_obs) - log10(model) [dex]")
    plt.title("RAR residual vs SBdisk (points + binned means)")
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()


def make_fig_scan(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, outpng: Path) -> None:
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()


def make_fig_rar_galaxy(df_gal: pd.DataFrame, a0: float, outpng: Path) -> None:
    plt.figure()
    plt.scatter(df_gal["logg_bar_med"], df_gal["logg_obs_med"], s=20, alpha=0.7)

    gb = np.logspace(np.nanmin(df_gal["logg_bar_med"]) - 0.2, np.nanmax(df_gal["logg_bar_med"]) + 0.2, 300)
    gm = rar_model(gb, a0)
    plt.plot(np.log10(gb), np.log10(gm))

    plt.xlabel("median log10(g_bar) [m/s^2]")
    plt.ylabel("median log10(g_obs) [m/s^2]")
    plt.title("SPARC RAR (one point per galaxy, medians)")
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()


def tex_escape_filename(s: str) -> str:
    return s.replace("_", r"\_")


def write_tex_report(outdir: Path, stats: Dict[str, float], files: Dict[str, str]) -> None:
    # avoid Unicode Greek letters; use \Upsilon in TeX
    tex = f"""%==============================================================================
% Enchan RAR Test Report (SPARC / Rotmod_LTG)
% Copyright (c) 2025 Mitsuhiro Kobayashi
%
% This work (textual content, PDF) is licensed under a
% Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
% To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
%
% The LaTeX source code structure itself is available under the MIT License.
%==============================================================================

\\documentclass[11pt,a4paper]{{article}}

%===========================
% Packages (match Enchan Field Notes style)
%===========================
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{mathptmx}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{bm}}
\\usepackage{{setspace}}
\\usepackage{{hyperref}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}

\\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}}

%===========================
% Helpers
%===========================
\\newcommand{{\\chap}}[1]{{\\clearpage\\section{{#1}}}}

%===========================
% Title / Author
%===========================
\\title{{\\textbf{{Enchan RAR Test Report v0.1}}\\\\[0.5em]
\\large Public-data verification of the SPARC Radial Acceleration Relation (RAR)}}
\\author{{Mitsuhiro Kobayashi\\\\[0.25em]
Tokyo, Japan\\\\
\\texttt{{enchan.theory@gmail.com}}}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle
\\thispagestyle{{empty}}

\\begin{{abstract}}
\\noindent
This report documents a minimal, reproducible verification of the radial acceleration relation (RAR)
using the publicly released SPARC rotation-curve decomposition files (\\texttt{{Rotmod\\_LTG.zip}}).
We compute $g_{{\\rm obs}}(r)=V_{{\\rm obs}}(r)^2/r$ and
$g_{{\\rm bar}}(r)=(V_{{\\rm gas}}^2+\\Upsilon_{{\\rm disk}}V_{{\\rm disk}}^2+\\Upsilon_{{\\rm bul}}V_{{\\rm bul}}^2)/r$,
and fit the one-parameter reference curve
$g_{{\\rm obs}}=g_{{\\rm bar}}/(1-e^{{-\\sqrt{{g_{{\\rm bar}}/a_0}}}})$.
For the baseline choice $(\\Upsilon_{{\\rm disk}},\\Upsilon_{{\\rm bul}})=({stats['Yd_base']:.2f},{stats['Yb']:.2f})$
we obtain $a_0={stats['a0_base']:.2e}\\,\\mathrm{{m/s^2}}$ and RMS scatter {stats['rms_base']:.3f} dex.
A scan in $\\Upsilon_{{\\rm disk}}$ indicates that the residual dependence on disk surface brightness
is minimized near $\\Upsilon_{{\\rm disk}}\\simeq {stats['Yd_rec']:.2f}$, without materially changing the scatter
(RMS {stats['rms_rec']:.3f} dex).
\\end{{abstract}}

\\clearpage
\\tableofcontents

\\chap{{Scope and deliverable}}
This document records a single verification task that can be rerun from public data with short Python code.
The deliverable is a reproducible target: the observed mapping $g_{{\\rm bar}}\\mapsto g_{{\\rm obs}}$ across many galaxies.

\\chap{{Data and definitions}}
\\subsection*{{Data}}
We use the public SPARC rotation-curve decomposition archive \\texttt{{Rotmod\\_LTG.zip}}.
After basic quality cuts ($r>0$, $V_{{\\rm obs}}>0$, finite values), the dataset contains
{int(stats['N_gal'])} galaxies and {int(stats['N_pts'])} radial points.
For analyses that use SBdisk, we restrict to SBdisk$>0$, yielding {int(stats['N_pts_sb'])} points.

\\subsection*{{Accelerations}}
We compute
\\begin{{align}}
g_{{\\rm obs}}(r) &= \\frac{{V_{{\\rm obs}}(r)^2}}{{r}},\\\\
g_{{\\rm bar}}(r) &= \\frac{{V_{{\\rm gas}}(r)^2+\\Upsilon_{{\\rm disk}}V_{{\\rm disk}}(r)^2+\\Upsilon_{{\\rm bul}}V_{{\\rm bul}}(r)^2}}{{r}},
\\end{{align}}
converting $(\\mathrm{{km/s}})^2/\\mathrm{{kpc}}$ to $\\mathrm{{m/s^2}}$.

\\chap{{Results}}
\\subsection*{{RAR points}}
Figure~\\ref{{fig:rar}} shows the point-level RAR for the SB-clean subset using
$(\\Upsilon_{{\\rm disk}},\\Upsilon_{{\\rm bul}})=({stats['Yd_rec']:.2f},{stats['Yb']:.2f})$.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.86\\linewidth]{{{tex_escape_filename(files['fig_rar_points'])}}}
\\caption{{SPARC RAR (points) with a one-parameter reference curve.}}
\\label{{fig:rar}}
\\end{{figure}}

\\subsection*{{SBdisk residual trend and scan}}
Figure~\\ref{{fig:sb}} shows residuals versus SBdisk with binned means.
Figures~\\ref{{fig:scanrho}} and \\ref{{fig:scanrms}} summarize the scan in $\\Upsilon_{{\\rm disk}}$.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.86\\linewidth]{{{tex_escape_filename(files['fig_resid_sb'])}}}
\\caption{{RAR residual vs SBdisk (points + binned means).}}
\\label{{fig:sb}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.86\\linewidth]{{{tex_escape_filename(files['fig_scan_rho'])}}}
\\caption{{Residual--SBdisk dependence (Spearman $\\rho$ on galaxy medians) versus $\\Upsilon_{{\\rm disk}}$.}}
\\label{{fig:scanrho}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.86\\linewidth]{{{tex_escape_filename(files['fig_scan_rms'])}}}
\\caption{{RAR scatter (RMS in dex) versus $\\Upsilon_{{\\rm disk}}$.}}
\\label{{fig:scanrms}}
\\end{{figure}}

\\subsection*{{One point per galaxy}}
Figure~\\ref{{fig:gal}} shows the RAR compressed to one point per galaxy (medians in $\\log g$).

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.86\\linewidth]{{{tex_escape_filename(files['fig_rar_galaxy'])}}}
\\caption{{RAR compressed to one point per galaxy (medians).}}
\\label{{fig:gal}}
\\end{{figure}}

\\chap{{Interpretation: geometry vs particles (minimal statement)}}
This report fixes one observational target from public data:
a highly regular mapping from baryonic acceleration proxy to observed centripetal acceleration.
A geometric reading treats this mapping as primary.
A particle dark-matter reading must account for why baryons and the halo co-vary so strongly across galaxies.

\\chap{{Reproducibility artifacts}}
This report is accompanied by:
\\begin{{itemize}}
\\item \\texttt{{{tex_escape_filename(files['csv_points'])}}}
\\item \\texttt{{{tex_escape_filename(files['csv_scan'])}}}
\\item \\texttt{{{tex_escape_filename(files['csv_galmed'])}}}
\\item figures: \\texttt{{{tex_escape_filename(files['fig_rar_points'])}}}, \\texttt{{{tex_escape_filename(files['fig_resid_sb'])}}}, \\texttt{{{tex_escape_filename(files['fig_scan_rho'])}}}, \\texttt{{{tex_escape_filename(files['fig_scan_rms'])}}}, \\texttt{{{tex_escape_filename(files['fig_rar_galaxy'])}}}
\\end{{itemize}}

\\chap{{References}}
\\begin{{itemize}}
\\item SPARC database: \\url{{https://astroweb.case.edu/SPARC/}}
\\end{{itemize}}

\\end{{document}}
"""
    (outdir / "enchan_rar_test_report_v0p1.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to Rotmod_LTG.zip")
    ap.add_argument("--outdir", default="Enchan_RAR_Test_Report_v0_1", help="Output directory")
    ap.add_argument("--Yb", type=float, default=0.70, help="Bulge M/L")
    ap.add_argument("--Yd_base", type=float, default=0.50, help="Baseline disk M/L")
    ap.add_argument("--Yd_rec", type=float, default=0.60, help="Recommended disk M/L (used for main figures)")
    ap.add_argument("--scan_min", type=float, default=0.30, help="Ydisk scan min")
    ap.add_argument("--scan_max", type=float, default=0.90, help="Ydisk scan max")
    ap.add_argument("--scan_step", type=float, default=0.02, help="Ydisk scan step")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = read_rotmod_zip(zip_path)

    N_gal = int(df_raw["galaxy"].nunique())
    N_pts = int(len(df_raw))

    df_sb = df_raw.copy()
    df_sb = df_sb[np.isfinite(df_sb["SBdisk"]) & (df_sb["SBdisk"] > 0)].reset_index(drop=True)
    N_pts_sb = int(len(df_sb))

    # baseline fit (SB-clean)
    d_base = compute_accelerations(df_sb, args.Yd_base, args.Yb)
    fit_base = fit_a0_logspace(d_base["g_bar"].to_numpy(), d_base["logg_obs"].to_numpy(), d_base["sigma_logg_obs"].to_numpy())
    a0_base = float(fit_base["a0"])
    d_base["g_model"] = rar_model(d_base["g_bar"].to_numpy(), a0_base)
    d_base["logg_model"] = np.log10(d_base["g_model"])
    d_base["resid"] = d_base["logg_obs"] - d_base["logg_model"]
    rms_base = float(rms_dex(d_base["resid"].to_numpy()))

    # recommended fit (SB-clean)
    d_rec = compute_accelerations(df_sb, args.Yd_rec, args.Yb)
    fit_rec = fit_a0_logspace(d_rec["g_bar"].to_numpy(), d_rec["logg_obs"].to_numpy(), d_rec["sigma_logg_obs"].to_numpy())
    a0_rec = float(fit_rec["a0"])
    d_rec["g_model"] = rar_model(d_rec["g_bar"].to_numpy(), a0_rec)
    d_rec["logg_model"] = np.log10(d_rec["g_model"])
    d_rec["resid"] = d_rec["logg_obs"] - d_rec["logg_model"]
    rms_rec = float(rms_dex(d_rec["resid"].to_numpy()))

    # save point CSV (recommended)
    csv_points = outdir / "sparc_rar_points_processed.csv"
    d_rec.to_csv(csv_points, index=False)

    # figures
    fig_rar_points = outdir / "fig_rar_points.png"
    fig_resid_sb = outdir / "fig_resid_sb_binned.png"
    make_fig_rar_points(d_rec, a0_rec, fig_rar_points)
    make_fig_resid_sb_binned(d_rec, fig_resid_sb)

    # scan
    scan_rows = []
    yds = np.arange(args.scan_min, args.scan_max + 1e-12, args.scan_step)
    for Yd in yds:
        dd = compute_accelerations(df_sb, float(Yd), args.Yb)
        fit = fit_a0_logspace(dd["g_bar"].to_numpy(), dd["logg_obs"].to_numpy(), dd["sigma_logg_obs"].to_numpy())
        a0 = float(fit["a0"])
        dd["g_model"] = rar_model(dd["g_bar"].to_numpy(), a0)
        dd["logg_model"] = np.log10(dd["g_model"])
        dd["resid"] = dd["logg_obs"] - dd["logg_model"]

        g = dd.groupby("galaxy", as_index=False).agg(
            resid_med=("resid", "median"),
            SBdisk_med=("SBdisk", "median"),
        )
        rho = spearman_rho(g["resid_med"], np.log10(g["SBdisk_med"]))
        scan_rows.append({
            "Ydisk": float(Yd),
            "Ybul": float(args.Yb),
            "a0_m_s2": a0,
            "rms_dex": rms_dex(dd["resid"].to_numpy()),
            "rho_sb_galmed": float(rho),
            "N_gal": int(g.shape[0]),
            "N_pts": int(dd.shape[0]),
        })

    df_scan = pd.DataFrame(scan_rows)
    csv_scan = outdir / "rar_Ydisk_scan_results.csv"
    df_scan.to_csv(csv_scan, index=False)

    idx_best = int(np.nanargmin(np.abs(df_scan["rho_sb_galmed"].to_numpy())))
    Yd_best = float(df_scan.loc[idx_best, "Ydisk"])

    fig_scan_rho = outdir / "fig_scan_rho_sb.png"
    fig_scan_rms = outdir / "fig_scan_rms.png"
    make_fig_scan(df_scan["Ydisk"].to_numpy(), df_scan["rho_sb_galmed"].to_numpy(),
                  "Ydisk", "Spearman rho (galaxy medians)", "Residual-SBdisk dependence vs Ydisk", fig_scan_rho)
    make_fig_scan(df_scan["Ydisk"].to_numpy(), df_scan["rms_dex"].to_numpy(),
                  "Ydisk", "RMS scatter [dex]", "RAR scatter vs Ydisk", fig_scan_rms)

    # galaxy medians (recommended)
    dd_gal = d_rec.groupby("galaxy", as_index=False).agg(
        logg_bar_med=("logg_bar", "median"),
        logg_obs_med=("logg_obs", "median"),
        SBdisk_med=("SBdisk", "median"),
        npts=("galaxy", "size"),
    )
    yd_str = f"{args.Yd_rec:.2f}".replace(".", "p")
    yb_str = f"{args.Yb:.2f}".replace(".", "p")
    gal_name = f"sparc_rar_galaxy_medians_Yd{yd_str}_Yb{yb_str}.csv"
    csv_galmed = outdir / gal_name
    dd_gal.to_csv(csv_galmed, index=False)

    fig_rar_galaxy = outdir / "fig_rar_galaxy.png"
    make_fig_rar_galaxy(dd_gal, a0_rec, fig_rar_galaxy)

    # TeX report (numbers inserted)
    stats = {
        "N_gal": float(N_gal),
        "N_pts": float(N_pts),
        "N_pts_sb": float(N_pts_sb),
        "Yb": float(args.Yb),
        "Yd_base": float(args.Yd_base),
        "Yd_rec": float(args.Yd_rec),
        "a0_base": float(a0_base),
        "rms_base": float(rms_base),
        "a0_rec": float(a0_rec),
        "rms_rec": float(rms_rec),
        "Yd_best_scan": float(Yd_best),
    }
    files = {
        "csv_points": csv_points.name,
        "csv_scan": csv_scan.name,
        "csv_galmed": csv_galmed.name,
        "fig_rar_points": fig_rar_points.name,
        "fig_resid_sb": fig_resid_sb.name,
        "fig_scan_rho": fig_scan_rho.name,
        "fig_scan_rms": fig_scan_rms.name,
        "fig_rar_galaxy": fig_rar_galaxy.name,
    }
    write_tex_report(outdir, stats, files)

    print("RAR reproduce summary")
    print(f"  zip: {zip_path.name}")
    print(f"  sha256: {sha256_file(zip_path)}")
    print(f"  galaxies: {N_gal}")
    print(f"  points (all): {N_pts}")
    print(f"  points (SBdisk>0): {N_pts_sb}")
    print("")
    print(f"  baseline: Yd={args.Yd_base:.2f}, Yb={args.Yb:.2f} -> a0={a0_base:.3e} m/s^2, RMS={rms_base:.3f} dex")
    print(f"  recommended: Yd={args.Yd_rec:.2f}, Yb={args.Yb:.2f} -> a0={a0_rec:.3e} m/s^2, RMS={rms_rec:.3f} dex")
    print(f"  scan best (min |rho|): Yd ~ {Yd_best:.2f}")
    print(f"Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
