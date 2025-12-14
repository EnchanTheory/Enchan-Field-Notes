#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan BTFR reproduce (SPARC BTFR .mrt)

Baseline / benchmark tool (non-Enchan):
- Parse a CDS-style .mrt fixed-width table (byte-by-byte column specs embedded in header)
- Extract one galaxy-level point per object: (log10 Mb, log10 Vf)
- Fit y = a + b x in log-log space (weighted in y if elogMb exists)
- Compute per-galaxy a0 implied by BTFR: a0_i = Vf^4 / (G * Mb)
- Output CSV tables, figures, and a minimal TeX stub (Enchan Field Notes style)

Usage:
  python enchan_btfr_reproduce.py --mrt BTFR_Lelli2019.mrt
  python enchan_btfr_reproduce.py --mrt BTFR_Lelli2019.mrt --max_elogMb 0.10
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Physical constants
G_SI = 6.67430e-11  # m^3 kg^-1 s^-2 (CODATA 2018; sufficient for this benchmark)
M_SUN = 1.98847e30  # kg
KM = 1000.0         # m


@dataclass
class ColSpec:
    start: int  # 0-based inclusive
    end: int    # 0-based exclusive
    label: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_label(s: str) -> str:
    """
    Make a safe column name from a CDS label.
    Examples:
      log(Mb) -> log_Mb
      e_log(Mb) -> e_log_Mb
    """
    s = s.strip()
    s = s.replace("\u2212", "-")  # unicode minus
    # preserve leading "e_" patterns and log() patterns reasonably
    s = s.replace("(", "_").replace(")", "")
    s = s.replace("/", "_per_")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def parse_mrt_fixedwidth(mrt_path: Path) -> pd.DataFrame:
    """Parse a CDS-style .mrt fixed-width table into a DataFrame.

    The parser uses the embedded "Byte-by-byte Description of file" block
    to build fixed-width column specs, then reads the data section.

    NOTE:
      Some SPARC .mrt tables (e.g., BTFR_Lelli2019.mrt) have data lines that are
      *shorter* than the maximum byte range in the spec block because trailing
      blanks are not present in the text file. Therefore, we must NOT require
      a minimum line length when detecting the first data row.
    """
    text = mrt_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Byte-by-byte spec: "1- 15  A15  ---  Name"
    spec_re = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s+\S+\s+\S+\s+(\S+)\s+")

    colspecs: List[ColSpec] = []
    in_spec_block = False
    spec_end_idx: Optional[int] = None

    for i, ln in enumerate(lines):
        if "Byte-by-byte Description of file" in ln:
            in_spec_block = True
            continue
        if not in_spec_block:
            continue

        # End of spec block (dashed separator) is common in CDS .mrt files
        if re.match(r"^\s*-{5,}\s*$", ln) and colspecs:
            spec_end_idx = i
            break

        m = spec_re.match(ln)
        if not m:
            continue
        b0 = int(m.group(1))
        b1 = int(m.group(2))
        label = _clean_label(m.group(3))
        colspecs.append(ColSpec(start=b0 - 1, end=b1, label=label))

    if not colspecs:
        raise RuntimeError("Could not find byte-by-byte column specs in the .mrt file header.")

    if spec_end_idx is None:
        # Fallback: if no dashed separator, start searching after last spec line
        spec_end_idx = 0
        for i, ln in enumerate(lines):
            if spec_re.match(ln):
                spec_end_idx = i

    dash_re = re.compile(r"^\s*-{5,}\s*$")
    skip_prefixes = ("Title:", "Authors:", "Abstract:", "Byte-by-byte", "Bytes", "Note")

    # Find the first data-like line after the spec block.
    data_start: Optional[int] = None
    for i in range(spec_end_idx + 1, len(lines)):
        ln = lines[i]
        if not ln.strip():
            continue
        if dash_re.match(ln):
            continue
        if ln.lstrip().startswith(skip_prefixes):
            continue
        # Data lines almost always contain digits and at least 2 tokens.
        # (We do NOT require a minimum character length.)
        if re.search(r"\d", ln) and len(ln.split()) >= 2:
            data_start = i
            break

    if data_start is None:
        # Last-resort: scan entire file
        for i, ln in enumerate(lines):
            if not ln.strip():
                continue
            if dash_re.match(ln):
                continue
            if ln.lstrip().startswith(skip_prefixes):
                continue
            if re.search(r"\d", ln) and len(ln.split()) >= 2:
                data_start = i
                break

    if data_start is None:
        raise RuntimeError("Could not locate the start of the data section in the .mrt file.")

    data_lines = lines[data_start:]
    colspec_tuples = [(c.start, c.end) for c in colspecs]
    names = [c.label for c in colspecs]

    df = pd.read_fwf(StringIO("\n".join(data_lines)), colspecs=colspec_tuples, names=names)
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    cols_map = {c.lower(): c for c in cols}

    for cand in candidates:
        c = cols_map.get(cand.lower())
        if c is not None:
            return c

    # fallback: substring match
    for cand in candidates:
        cl = cand.lower()
        for c in cols:
            if cl in c.lower():
                return c
    return None


def extract_btfr_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract:
      name, logMb (dex Msun), elogMb (dex), logVf (dex km/s), elogVf (dex)
    Robust to common SPARC MRT naming variants.
    """
    name_col = _find_col(df_raw, ["Name", "Galaxy", "ID", "Gal", "Object"]) or df_raw.columns[0]

    # mass columns
    logMb_col = _find_col(df_raw, ["log_Mb", "logMb", "log_Mbar", "logMbar", "log_Mbary", "logMbary", "log_Mb_"])
    Mb_col = _find_col(df_raw, ["Mb", "Mbar", "Mbary", "M_b"])
    elogMb_col = _find_col(df_raw, ["e_log_Mb", "e_logMb", "elogMb", "e_log_Mbar", "e_logMbar", "e_log_Mb_"])
    eMb_col = _find_col(df_raw, ["e_Mb", "eMb", "e_Mbar", "eMbar"])

    # velocity columns
    logVf_col = _find_col(df_raw, ["log_Vf", "logVf", "log_Vflat", "logVflat"])
    Vf_col = _find_col(df_raw, ["Vf", "Vflat", "V_f", "V_flat"])
    elogVf_col = _find_col(df_raw, ["e_log_Vf", "e_logVf", "elogVf", "e_log_Vflat", "e_logVflat"])
    eVf_col = _find_col(df_raw, ["e_Vf", "eVf", "e_Vflat", "eVflat"])

    out = pd.DataFrame()
    out["name"] = df_raw[name_col].astype(str).str.strip()

    # logMb
    if logMb_col is not None:
        out["logMb"] = pd.to_numeric(df_raw[logMb_col], errors="coerce")
        if elogMb_col is not None:
            out["elogMb"] = pd.to_numeric(df_raw[elogMb_col], errors="coerce")
        elif eMb_col is not None:
            eMb = pd.to_numeric(df_raw[eMb_col], errors="coerce")
            Mb = np.power(10.0, out["logMb"])
            out["elogMb"] = eMb / (Mb * np.log(10.0))
        else:
            out["elogMb"] = np.nan
    elif Mb_col is not None:
        Mb = pd.to_numeric(df_raw[Mb_col], errors="coerce")
        out["logMb"] = np.log10(Mb)
        if eMb_col is not None:
            eMb = pd.to_numeric(df_raw[eMb_col], errors="coerce")
            out["elogMb"] = eMb / (Mb * np.log(10.0))
        else:
            out["elogMb"] = np.nan
    else:
        raise RuntimeError("Could not find Mb or log(Mb) columns in the .mrt file.")

    # logVf
    if logVf_col is not None:
        out["logVf"] = pd.to_numeric(df_raw[logVf_col], errors="coerce")
        if elogVf_col is not None:
            out["elogVf"] = pd.to_numeric(df_raw[elogVf_col], errors="coerce")
        elif eVf_col is not None:
            eVf = pd.to_numeric(df_raw[eVf_col], errors="coerce")
            Vf = np.power(10.0, out["logVf"])
            out["elogVf"] = eVf / (Vf * np.log(10.0))
        else:
            out["elogVf"] = np.nan
    elif Vf_col is not None:
        Vf = pd.to_numeric(df_raw[Vf_col], errors="coerce")
        out["logVf"] = np.log10(Vf)
        if eVf_col is not None:
            eVf = pd.to_numeric(df_raw[eVf_col], errors="coerce")
            out["elogVf"] = eVf / (Vf * np.log(10.0))
        else:
            out["elogVf"] = np.nan
    else:
        raise RuntimeError("Could not find Vf or log(Vf) columns in the .mrt file.")

    return out


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, sy: np.ndarray) -> Tuple[float, float]:
    """
    Weighted least squares fit of y = a + b x.
    If sy is missing/invalid, fall back to equal weights.
    """
    w = np.ones_like(y, dtype=float)
    if np.isfinite(sy).any():
        w = 1.0 / np.where(np.isfinite(sy) & (sy > 0), sy**2, np.nan)

        finite_w = w[np.isfinite(w)]
        if finite_w.size > 0:
            fill = float(np.nanmedian(finite_w))
            w = np.where(np.isfinite(w), w, fill)
        else:
            w = np.ones_like(y, dtype=float)

    W = np.sum(w)
    xw = np.sum(w * x) / W
    yw = np.sum(w * y) / W
    Sxx = np.sum(w * (x - xw) ** 2)
    Sxy = np.sum(w * (x - xw) * (y - yw))
    b = Sxy / Sxx
    a = yw - b * xw
    return float(a), float(b)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def percentile_summary(v: np.ndarray, p_lo: float = 16.0, p_hi: float = 84.0) -> Tuple[float, float, float]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.nanmedian(v))
    lo = float(np.nanpercentile(v, p_lo))
    hi = float(np.nanpercentile(v, p_hi))
    return med, lo, hi


def make_report_tex(outdir: Path, mrt_name: str, stats: Dict[str, float]) -> None:
    """
    Minimal TeX stub that matches Enchan Field Notes header style.
    """
    tex = f"""%==============================================================================
% Enchan BTFR Test Report (SPARC BTFR table)
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

\\newcommand{{\\chap}}[1]{{\\clearpage\\section{{#1}}}}

\\title{{\\textbf{{Enchan BTFR Test Report v0.1}}\\\\[0.5em]
\\large One-point-per-galaxy verification from a public SPARC BTFR table}}
\\author{{Mitsuhiro Kobayashi\\\\[0.25em]
Tokyo, Japan\\\\
\\texttt{{enchan.theory@gmail.com}}}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle
\\thispagestyle{{empty}}

\\begin{{abstract}}
\\noindent
This report documents a minimal, reproducible BTFR benchmark using the public SPARC BTFR table
\\texttt{{{mrt_name}}}. After basic finite-value cuts ($N={int(stats['N'])}$), we fit
$\\log_{{10}} M_{{\\rm b}} = a + b\\,\\log_{{10}} V_{{\\rm f}}$ and obtain
$a={stats['a']:.3f}$, $b={stats['b']:.3f}$ with RMS scatter {stats['rms']:.3f} dex in $\\log_{{10}} M_{{\\rm b}}$.
We also compute the implied acceleration scale per galaxy
$a_0 = V_{{\\rm f}}^4 / (G M_{{\\rm b}})$ and report a median
$a_0={stats['a0_med']:.3e}\\,\\mathrm{{m/s^2}}$ (16--84\\%: {stats['a0_p16']:.3e}--{stats['a0_p84']:.3e}).
\\end{{abstract}}

\\clearpage
\\tableofcontents

\\chap{{Scope and deliverable}}
This document records a single verification task that can be rerun from public data with short Python code.
It fixes a transparent extraction of one point per galaxy and provides a benchmark fit and a derived
$a_0$ distribution.

\\chap{{Results}}
\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.88\\linewidth]{{fig_btfr_points.png}}
\\caption{{BTFR points and best-fit line.}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.88\\linewidth]{{fig_btfr_residuals.png}}
\\caption{{Residuals in $\\log_{{10}} M_{{\\rm b}}$ around the best-fit line.}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.88\\linewidth]{{fig_a0_hist.png}}
\\caption{{Distribution of $a_0 = V_{{\\rm f}}^4/(G M_{{\\rm b}})$ implied by the BTFR table.}}
\\end{{figure}}

\\chap{{Reproducibility artifacts}}
\\begin{{itemize}}
\\item \\texttt{{btfr_points_processed.csv}}
\\item \\texttt{{btfr_fit_summary.csv}}
\\item \\texttt{{btfr_a0_summary.csv}}
\\item \\texttt{{fig_btfr_points.png}}, \\texttt{{fig_btfr_residuals.png}}, \\texttt{{fig_a0_hist.png}}
\\end{{itemize}}

\\chap{{References}}
\\begin{{itemize}}
\\item SPARC database: \\url{{https://astroweb.case.edu/SPARC/}}
\\end{{itemize}}

\\end{{document}}
"""
    (outdir / "enchan_btfr_test_report_v0p1.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrt", required=True, help="Path to SPARC BTFR .mrt table (e.g., BTFR_Lelli2019.mrt)")
    ap.add_argument("--outdir", default="Enchan_BTFR_Test_Report_v0_1", help="Output directory")
    ap.add_argument("--max_elogMb", type=float, default=None,
                    help="Optional quality cut: keep rows with elogMb <= threshold (dex) if elogMb exists")
    ap.add_argument("--make-tex", action="store_true", help="Write a minimal TeX report stub into outdir")
    args = ap.parse_args()

    mrt_path = Path(args.mrt)
    if not mrt_path.exists():
        raise FileNotFoundError(f"Missing input file: {mrt_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mrt_sha = sha256_file(mrt_path)

    df_raw = parse_mrt_fixedwidth(mrt_path)
    df = extract_btfr_columns(df_raw)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["logMb", "logVf"]).reset_index(drop=True)

    # Optional cut on elogMb
    if args.max_elogMb is not None and "elogMb" in df.columns:
        df = df[(df["elogMb"].isna()) | (df["elogMb"] <= args.max_elogMb)].reset_index(drop=True)

    # Hard cut: Vf must be positive (logVf finite already implies Vf>0, but keep explicit)
    Vf_kms = np.power(10.0, df["logVf"].to_numpy(dtype=float))
    df = df[np.isfinite(Vf_kms) & (Vf_kms > 0)].reset_index(drop=True)

    N = len(df)
    if N < 10:
        raise RuntimeError(f"Too few rows after cuts (N={N}).")

    x = df["logVf"].to_numpy(dtype=float)  # log10(km/s)
    y = df["logMb"].to_numpy(dtype=float)  # log10(Msun)
    sy = df["elogMb"].to_numpy(dtype=float) if "elogMb" in df.columns else np.full_like(y, np.nan)

    a, b = weighted_linear_fit(x, y, sy)
    yhat = a + b * x
    resid = y - yhat
    rms_y = rms(resid)

    # a0 per galaxy: a0 = Vf^4 / (G * Mb)
    Mb_kg = np.power(10.0, y) * M_SUN
    Vf_mps = np.power(10.0, x) * KM
    a0_i = (Vf_mps ** 4) / (G_SI * Mb_kg)
    loga0_i = np.log10(a0_i)

    # Summaries
    a0_med, a0_p16, a0_p84 = percentile_summary(a0_i)
    loga0_med, loga0_p16, loga0_p84 = percentile_summary(loga0_i)

    # Store processed rows
    df_out = df.copy()
    df_out["yhat"] = yhat
    df_out["resid_logMb"] = resid
    df_out["Vf_km_s"] = Vf_kms[:len(df_out)]
    df_out["Mb_Msun"] = np.power(10.0, y)
    df_out["a0_m_s2"] = a0_i
    df_out["log10_a0_m_s2"] = loga0_i
    df_out.to_csv(outdir / "btfr_points_processed.csv", index=False)

    # Fit summary
    fit_summary = pd.DataFrame([{
        "N": N,
        "a": a,
        "b": b,
        "rms_dex_logMb": rms_y,
        "mrt_file": mrt_path.name,
        "sha256": mrt_sha,
        "quality_cut_max_elogMb": (args.max_elogMb if args.max_elogMb is not None else "")
    }])
    fit_summary.to_csv(outdir / "btfr_fit_summary.csv", index=False)

    # a0 summary
    a0_summary = pd.DataFrame([{
        "N": N,
        "a0_median_m_s2": a0_med,
        "a0_p16_m_s2": a0_p16,
        "a0_p84_m_s2": a0_p84,
        "log10_a0_median": loga0_med,
        "log10_a0_p16": loga0_p16,
        "log10_a0_p84": loga0_p84,
        "mrt_file": mrt_path.name,
        "sha256": mrt_sha
    }])
    a0_summary.to_csv(outdir / "btfr_a0_summary.csv", index=False)

    # Figures
    plt.figure()
    plt.scatter(x, y, s=12, alpha=0.7)
    xs = np.linspace(np.nanmin(x) - 0.05, np.nanmax(x) + 0.05, 200)
    plt.plot(xs, a + b * xs)
    plt.xlabel("log10(Vf [km/s])")
    plt.ylabel("log10(Mb [Msun])")
    plt.title("SPARC BTFR (one point per galaxy)")
    plt.tight_layout()
    plt.savefig(outdir / "fig_btfr_points.png", dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(x, resid, s=12, alpha=0.7)
    plt.axhline(0.0)
    plt.xlabel("log10(Vf [km/s])")
    plt.ylabel("Residual in log10(Mb) [dex]")
    plt.title("BTFR residuals around best-fit line")
    plt.tight_layout()
    plt.savefig(outdir / "fig_btfr_residuals.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(loga0_i[np.isfinite(loga0_i)], bins=30)
    plt.axvline(loga0_med)
    plt.xlabel("log10(a0 [m/s^2])")
    plt.ylabel("Count")
    plt.title("a0 implied by BTFR: a0 = Vf^4 / (G Mb)")
    plt.tight_layout()
    plt.savefig(outdir / "fig_a0_hist.png", dpi=200)
    plt.close()

    if args.make_tex:
        make_report_tex(outdir, mrt_path.name, {
            "N": float(N),
            "a": float(a),
            "b": float(b),
            "rms": float(rms_y),
            "a0_med": float(a0_med),
            "a0_p16": float(a0_p16),
            "a0_p84": float(a0_p84),
        })

    # Console summary (compact)
    print("BTFR reproduce summary")
    print(f"  mrt: {mrt_path.name}")
    print(f"  sha256: {mrt_sha}")
    print(f"  galaxies (N): {N}")
    print(f"  fit: a={a:.3f}, b={b:.3f}, RMS={rms_y:.3f} dex")
    print(f"  a0(BTFR): median={a0_med:.3e} m/s^2 (16–84%: {a0_p16:.3e}–{a0_p84:.3e})")
    print(f"           log10 a0 median={loga0_med:.3f} (16–84%: {loga0_p16:.3f}–{loga0_p84:.3f})")
    print(f"Outputs written to: {outdir.resolve()}")
    print("Key files:")
    print("  - btfr_points_processed.csv")
    print("  - btfr_fit_summary.csv")
    print("  - btfr_a0_summary.csv")
    print("  - fig_btfr_points.png, fig_btfr_residuals.png, fig_a0_hist.png")
    if args.make_tex:
        print("  - enchan_btfr_test_report_v0p1.tex")


if __name__ == "__main__":
    main()
