#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan BTFR reproduce (SPARC BTFR .mrt table)

- Parses CDS-style .mrt (byte-by-byte column specs + fixed-width data)
- Extracts baryonic mass (Mb or log(Mb)) and flat velocity (Vf or log(Vf))
- Fits log10(Mb) = a + b*log10(Vf) (weighted in y if e_logMb present)
- Produces: processed CSV, summary CSV, figures, and a TeX report (Enchan Field Notes style)

No SciPy required. Dependencies: numpy, pandas, matplotlib.

Usage:
  python enchan_btfr_reproduce.py --mrt BTFR_Lelli2019.mrt
  python enchan_btfr_reproduce.py --mrt BTFR_Lelli2019.mrt --outdir Enchan_BTFR_Test_Report_v0_1
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tex_escape_texttt(s: str) -> str:
    return s.replace("\\", "/").replace("_", "\\_")


def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("−", "-")
    s = s.replace(" ", "")
    s = s.replace("(", "").replace(")", "")
    s = s.replace("/", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


@dataclass
class ColSpec:
    start: int
    end: int
    label: str


def parse_mrt_fixedwidth(mrt_path: Path) -> pd.DataFrame:
    """
    Parse a CDS .mrt file using the embedded Byte-by-byte table.

    SPARC BTFR tables typically have:
      - a "Byte-by-byte Description of file" block (column specs)
      - one or more dashed separators
      - a fixed-width data section (one row per galaxy)

    This parser:
      1) reads the byte-by-byte spec block to build colspecs
      2) locates the first plausible data line *after* the spec block
      3) uses pandas.read_fwf to load the fixed-width table
    """
    text = mrt_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    in_spec = False
    colspecs: List[ColSpec] = []
    max_end_1based = 0
    spec_end_idx: Optional[int] = None

    # Example spec line:
    #  1- 12  A12   ---   Name    Galaxy name
    spec_re = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+")

    for idx, ln in enumerate(lines):
        if "Byte-by-byte Description of file" in ln:
            in_spec = True
            continue
        if not in_spec:
            continue
        # spec block usually ends at a dashed line after at least one spec row
        if ln.strip().startswith("----") and colspecs:
            spec_end_idx = idx
            break
        m = spec_re.match(ln)
        if not m:
            continue
        b0 = int(m.group(1))
        b1 = int(m.group(2))
        label = m.group(5)
        start = b0 - 1
        end = b1
        max_end_1based = max(max_end_1based, b1)
        colspecs.append(ColSpec(start=start, end=end, label=label))

    if not colspecs:
        raise RuntimeError("Could not find byte-by-byte column specs in the .mrt file.")

    if spec_end_idx is None:
        spec_end_idx = 0

    # Find first plausible data line after the spec block.
    data_start_idx: Optional[int] = None
    for i in range(spec_end_idx + 1, len(lines)):
        ln = lines[i]
        if ln.strip() == "":
            continue
        # skip separators
        if set(ln.strip()) == {"-"}:
            continue
        head = ln.lstrip()
        if head.startswith(("Bytes", "Byte-by-byte", "Title:", "Authors:", "Table", "Note")):
            continue
        # skip spec-like rows
        if spec_re.match(ln):
            continue
        # must be wide enough for fixed-width parsing
        if len(ln) < (max_end_1based - 2):
            continue
        data_start_idx = i
        break

    if data_start_idx is None:
        raise RuntimeError("Could not locate the start of the data section in the .mrt file.")

    data_lines = lines[data_start_idx:]
    # Some CDS tables omit trailing spaces; pad to spec width for safe fixed-width parsing.
    pad_to = max_end_1based
    data_lines = [dl.ljust(pad_to) for dl in data_lines]
    colspec_tuples = [(c.start, c.end) for c in colspecs]
    names = [_norm(c.label) for c in colspecs]

    df = pd.read_fwf(StringIO("\n".join(data_lines)), colspecs=colspec_tuples, names=names)
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    nmap = {_norm(c): c for c in cols}
    for cand in candidates:
        key = _norm(cand)
        if key in nmap:
            return nmap[key]
    for cand in candidates:
        key = _norm(cand)
        for c in cols:
            if key in _norm(c):
                return c
    return None


def extract_btfr(df_raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    name_col = find_col(df_raw, ["name", "galaxy", "id", "obj", "object"]) or df_raw.columns[0]
    out["name"] = df_raw[name_col].astype(str).str.strip()

    logMb_col = find_col(df_raw, ["logmb", "log_mb", "logmbar", "log_mbar", "logmbary", "log_mbary"])
    Mb_col = find_col(df_raw, ["mb", "mbar", "mbary", "m_b"])
    elogMb_col = find_col(df_raw, ["e_logmb", "elogmb", "e_log_mbar", "elogmbar", "e_logmbary"])
    eMb_col = find_col(df_raw, ["e_mb", "emb", "e_mbar", "embar"])

    logVf_col = find_col(df_raw, ["logvf", "log_vf", "logvflat", "log_vflat"])
    Vf_col = find_col(df_raw, ["vf", "vflat", "v_f", "v_flat"])
    elogVf_col = find_col(df_raw, ["e_logvf", "elogvf", "e_logvflat", "elogvflat"])
    eVf_col = find_col(df_raw, ["e_vf", "evf", "e_vflat", "evflat"])

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
        Mb = Mb.where(Mb > 0)
        out["logMb"] = np.log10(Mb)
        if eMb_col is not None:
            eMb = pd.to_numeric(df_raw[eMb_col], errors="coerce")
            out["elogMb"] = eMb / (Mb * np.log(10.0))
        else:
            out["elogMb"] = np.nan
    else:
        raise RuntimeError("Could not find Mb or log(Mb) in the .mrt table.")

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
        Vf = Vf.where(Vf > 0)
        out["logVf"] = np.log10(Vf)
        if eVf_col is not None:
            eVf = pd.to_numeric(df_raw[eVf_col], errors="coerce")
            out["elogVf"] = eVf / (Vf * np.log(10.0))
        else:
            out["elogVf"] = np.nan
    else:
        raise RuntimeError("Could not find Vf or log(Vf) in the .mrt table.")

    return out


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, sy: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sy = np.asarray(sy, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        raise RuntimeError("Too few finite points for fit.")
    x = x[mask]
    y = y[mask]
    sy = sy[mask]

    w = np.ones_like(y)
    ok = np.isfinite(sy) & (sy > 0)
    if ok.any():
        w = np.where(ok, 1.0 / (sy**2), np.nan)
        finite = w[np.isfinite(w)]
        if finite.size > 0:
            w = np.where(np.isfinite(w), w, np.median(finite))
        else:
            w = np.ones_like(y)

    W = np.sum(w)
    xw = np.sum(w * x) / W
    yw = np.sum(w * y) / W
    Sxx = np.sum(w * (x - xw)**2)
    Sxy = np.sum(w * (x - xw) * (y - yw))
    b = Sxy / Sxx
    a = yw - b * xw
    return float(a), float(b)


def make_report_tex(outdir: Path, stats: Dict[str, float], mrt_name: str, mrt_sha256: str) -> None:
    tex_path = outdir / "enchan_btfr_test_report_v0p1.tex"
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
This report documents a minimal, reproducible verification of the baryonic Tully--Fisher relation (BTFR)
using a public SPARC BTFR table (\\texttt{{{tex_escape_texttt(mrt_name)}}}).
We extract one galaxy-level point per object: baryonic mass $M_{{\\rm b}}$ and flat rotation velocity $V_{{\\rm f}}$,
and fit a log--log power law of the form $\\log_{{10}} M_{{\\rm b}} = a + b\\,\\log_{{10}} V_{{\\rm f}}$.
For the selected sample ($N={int(stats['N'])}$), we obtain
$a={stats['a']:.3f}$, $b={stats['b']:.3f}$, and an RMS scatter of {stats['rms']:.3f} dex in $\\log_{{10}} M_{{\\rm b}}$.
\\end{{abstract}}

\\clearpage
\\tableofcontents

\\chap{{Scope and deliverable}}
This document is intentionally narrow: it records a single verification task that can be rerun from public data with short Python code.
The goal is to create a clean external handle on a core dark-matter symptom: one baryonic number per galaxy predicts one dynamical number per galaxy with small scatter.

\\chap{{Data and definitions}}
\\subsection*{{Input table}}
We parse a CDS-style fixed-width \\texttt{{.mrt}} file using its embedded byte-by-byte column specification.
Input: \\texttt{{{tex_escape_texttt(mrt_name)}}}.\\\\
SHA256: \\texttt{{{mrt_sha256}}}.

\\subsection*{{Quantities}}
We define
\\[
x \\equiv \\log_{{10}}\\left(V_{{\\rm f}}/\\mathrm{{km\\,s^{{-1}}}}\\right),\\qquad
y \\equiv \\log_{{10}}\\left(M_{{\\rm b}}/M_\\odot\\right).
\\]
In this benchmark we use the table-provided values for $M_{{\\rm b}}$ and $V_{{\\rm f}}$ (or their logarithms), and apply only basic finite-value cuts and $V_{{\\rm f}}>0$.

\\chap{{Fit and metrics}}
We fit $y=a+bx$ using weighted least squares in $y$ when an uncertainty column is available for $\\log_{{10}} M_{{\\rm b}}$.
We report RMS scatter in $y$ (dex) around the best-fit line.

\\chap{{Results}}
\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.88\\linewidth]{{fig_btfr_points.png}}
\\caption{{BTFR points and best-fit line in log--log space (one point per galaxy).}}
\\label{{fig:btfr}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.88\\linewidth]{{fig_btfr_residuals.png}}
\\caption{{Residuals in $\\log_{{10}} M_{{\\rm b}}$ around the best-fit line.}}
\\label{{fig:resid}}
\\end{{figure}}

\\begin{{table}}[htbp]
\\centering
\\begin{{tabular}}{{lcc}}
\\toprule
Metric & Value & Notes\\\\
\\midrule
Galaxies $N$ & {int(stats['N'])} & after basic cuts\\\\
Intercept $a$ & {stats['a']:.3f} & $\\log_{{10}} M_{{\\rm b}} = a + b\\log_{{10}} V_{{\\rm f}}$\\\\
Slope $b$ & {stats['b']:.3f} & log--log slope\\\\
RMS scatter & {stats['rms']:.3f} dex & in $\\log_{{10}} M_{{\\rm b}}$\\\\
\\bottomrule
\\end{{tabular}}
\\caption{{BTFR fit summary for the selected sample.}}
\\end{{table}}

\\chap{{Interpretation: geometry vs particles (minimal statement)}}
This report fixes a reproducible benchmark: a near power-law mapping between baryonic mass and flat rotation velocity is strongly present at the one-galaxy-one-point level.
This test does not select a unique theory; it provides a clean target that any explanation must reproduce.

\\chap{{Reproducibility artifacts}}
\\begin{{itemize}}
\\item \\texttt{{{tex_escape_texttt('btfr_points_processed.csv')}}}
\\item \\texttt{{{tex_escape_texttt('btfr_fit_summary.csv')}}}
\\item \\texttt{{{tex_escape_texttt('fig_btfr_points.png')}}}, \\texttt{{{tex_escape_texttt('fig_btfr_residuals.png')}}}
\\item \\texttt{{{tex_escape_texttt('enchan_btfr_test_report_v0p1.tex')}}}
\\end{{itemize}}

\\chap{{References}}
\\begin{{itemize}}
\\item SPARC database (downloads): \\url{{https://astroweb.case.edu/SPARC/}}
\\end{{itemize}}

\\end{{document}}
"""
    tex_path.write_text(tex, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrt", required=True, help="Path to SPARC BTFR .mrt table (e.g., BTFR_Lelli2019.mrt)")
    ap.add_argument("--outdir", default="Enchan_BTFR_Test_Report_v0_1", help="Output directory")
    ap.add_argument("--max_elogMb", type=float, default=None,
                    help="Optional cut: keep rows with elogMb <= value (dex) when elogMb exists")
    args = ap.parse_args()

    mrt_path = Path(args.mrt)
    if not mrt_path.exists():
        raise FileNotFoundError(f"Missing input file: {mrt_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mrt_sha = sha256_file(mrt_path)

    df_raw = parse_mrt_fixedwidth(mrt_path)
    df = extract_btfr(df_raw)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["logMb", "logVf"]).reset_index(drop=True)

    if args.max_elogMb is not None and "elogMb" in df.columns:
        df = df[df["elogMb"].isna() | (df["elogMb"] <= args.max_elogMb)].reset_index(drop=True)

    N = int(len(df))
    if N < 10:
        raise RuntimeError(f"Too few galaxies after cuts (N={N}).")

    x = df["logVf"].to_numpy(dtype=float)
    y = df["logMb"].to_numpy(dtype=float)
    sy = df["elogMb"].to_numpy(dtype=float) if "elogMb" in df.columns else np.full_like(y, np.nan)

    a, b = weighted_linear_fit(x, y, sy)
    yhat = a + b * x
    resid = y - yhat
    rms = float(np.sqrt(np.mean(resid**2)))
    r_pearson = float(np.corrcoef(x, y)[0, 1]) if np.isfinite(x).all() and np.isfinite(y).all() else float("nan")

    df["yhat"] = yhat
    df["resid"] = resid
    df.to_csv(outdir / "btfr_points_processed.csv", index=False)

    summary = pd.DataFrame([{
        "mrt_file": mrt_path.name,
        "mrt_sha256": mrt_sha,
        "N": N,
        "a": a,
        "b": b,
        "rms_dex_logMb": rms,
        "pearson_r_xy": r_pearson,
        "cut_max_elogMb": (args.max_elogMb if args.max_elogMb is not None else ""),
    }])
    summary.to_csv(outdir / "btfr_fit_summary.csv", index=False)

    plt.figure()
    plt.scatter(x, y, s=14, alpha=0.75)
    xs = np.linspace(np.nanmin(x) - 0.05, np.nanmax(x) + 0.05, 250)
    plt.plot(xs, a + b * xs)
    plt.xlabel("log10(Vf [km/s])")
    plt.ylabel("log10(Mb [Msun])")
    plt.title("SPARC BTFR (one point per galaxy)")
    plt.tight_layout()
    plt.savefig(outdir / "fig_btfr_points.png", dpi=220)
    plt.close()

    plt.figure()
    plt.scatter(x, resid, s=14, alpha=0.75)
    plt.axhline(0.0)
    plt.xlabel("log10(Vf [km/s])")
    plt.ylabel("Residual in log10(Mb) [dex]")
    plt.title("BTFR residuals around best-fit line")
    plt.tight_layout()
    plt.savefig(outdir / "fig_btfr_residuals.png", dpi=220)
    plt.close()

    make_report_tex(outdir, {"N": float(N), "a": a, "b": b, "rms": rms}, mrt_path.name, mrt_sha)

    print("BTFR reproduce summary")
    print(f"  mrt: {mrt_path.name}")
    print(f"  sha256: {mrt_sha}")
    print(f"  galaxies (N): {N}")
    print(f"  fit: a={a:.3f}, b={b:.3f}, RMS={rms:.3f} dex")
    print(f"Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()