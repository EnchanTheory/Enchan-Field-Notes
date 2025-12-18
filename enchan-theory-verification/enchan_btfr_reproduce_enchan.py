#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan BTFR reproduce (ENCHAN model)

- Parses CDS-style SPARC BTFR .mrt (fixed-width + embedded byte-by-byte specs)
- Extracts one point per galaxy: (Mb, Vf)
- Reports:
    (1) free-slope linear fit in log-log (benchmark summary)
    (2) Enchan/BTFR implied a0 per galaxy: a0 = Vf^4 / (G Mb)
    (3) a0 summary (median, 16th/84th) + histogram
    (4) fixed-slope test (b=4): best a0 (in least-squares sense) + RMS scatter

This is intended as the "a0 anchor" for Enchan RAR/rotation-curve prediction tests.

Usage:
  python enchan_btfr_reproduce_enchan.py --mrt BTFR_Lelli2019.mrt
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from enchan_core_model import G_SI, MSUN_KG, KMS_TO_MS, a0_from_btfr, robust_percentiles


@dataclass
class ColSpec:
    start: int  # 0-based inclusive
    end: int    # 0-based exclusive
    label: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_label(s: str) -> str:
    s = s.strip()
    s = s.replace("\u2212", "-")  # unicode minus
    # keep Mb/Vf tokens readable
    s = s.replace("(", "").replace(")", "")
    s = s.replace("/", "_per_")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def parse_mrt_fixedwidth(mrt_path: Path) -> pd.DataFrame:
    """
    Robust parser for CDS-style .mrt:
    - reads byte-by-byte specs from the header
    - then reads the fixed-width data section via pandas.read_fwf

    The "data start" heuristic is intentionally permissive:
    we accept the first non-header, non-rule line that contains at least one digit.
    """
    text = mrt_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    spec_re = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s+\S+\s+\S+\s+(\S+)\s+")
    colspecs: List[ColSpec] = []
    in_spec = False
    max_end_1based = 0

    for ln in lines:
        if "Byte-by-byte Description of file" in ln:
            in_spec = True
            continue
        if not in_spec:
            continue
        if ln.strip().startswith("----") and colspecs:
            # end of the spec block
            break
        m = spec_re.match(ln)
        if not m:
            continue
        b0 = int(m.group(1))
        b1 = int(m.group(2))
        label = _clean_label(m.group(3))
        colspecs.append(ColSpec(start=b0 - 1, end=b1, label=label))
        max_end_1based = max(max_end_1based, b1)

    if not colspecs:
        raise RuntimeError("Could not find byte-by-byte column specs in the .mrt file.")

    # Locate data section: first "row-like" line after header/rules.
    data_start_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        if ln.startswith("Title:") or ln.startswith("Authors:") or ln.startswith("Table:"):
            continue
        if "Byte-by-byte Description of file" in ln:
            continue
        if ln.strip().startswith("Bytes") or ln.strip().startswith("----") or ln.strip().startswith("==="):
            continue
        if ln.lstrip().startswith("#"):
            continue
        if re.search(r"\d", ln) and len(ln.split()) >= 2:
            data_start_idx = i
            break

    if data_start_idx is None:
        raise RuntimeError("Could not locate the start of the data section in the .mrt file.")

    data_lines = lines[data_start_idx:]
    colspec_tuples = [(c.start, c.end) for c in colspecs]
    names = [c.label for c in colspecs]
    df = pd.read_fwf(StringIO("\n".join(data_lines)), colspecs=colspec_tuples, names=names)
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # substring fallback
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None


def extract_btfr(df_raw: pd.DataFrame) -> pd.DataFrame:
    name_col = _find_col(df_raw, ["Name", "Galaxy", "ID", "Gal", "Object"]) or df_raw.columns[0]

    logMb_col = _find_col(df_raw, ["log_Mb", "logMb", "log_Mbar", "logMbar", "log_Mbary", "logMbary", "log_Mb_"])
    Mb_col = _find_col(df_raw, ["Mb", "Mbar", "Mbary", "M_b"])

    elogMb_col = _find_col(df_raw, ["e_log_Mb", "e_logMb", "elogMb", "e_log_Mbar", "e_logMbar"])
    eMb_col = _find_col(df_raw, ["e_Mb", "eMb", "e_Mbar", "eMbar"])

    logVf_col = _find_col(df_raw, ["log_Vf", "logVf", "log_Vflat", "logVflat"])
    Vf_col = _find_col(df_raw, ["Vf", "Vflat", "V_f", "V_flat"])

    elogVf_col = _find_col(df_raw, ["e_log_Vf", "e_logVf", "elogVf", "e_log_Vflat", "e_logVflat"])
    eVf_col = _find_col(df_raw, ["e_Vf", "eVf", "e_Vflat", "eVflat"])

    out = pd.DataFrame()
    out["name"] = df_raw[name_col].astype(str).str.strip()

    # Mb
    if logMb_col is not None:
        out["logMb"] = pd.to_numeric(df_raw[logMb_col], errors="coerce")
        if elogMb_col is not None:
            out["elogMb"] = pd.to_numeric(df_raw[elogMb_col], errors="coerce")
        elif eMb_col is not None:
            Mb = np.power(10.0, out["logMb"])
            eMb = pd.to_numeric(df_raw[eMb_col], errors="coerce")
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
        raise RuntimeError("Could not find Mb or log(Mb) columns in the .mrt file.")

    # Vf
    if logVf_col is not None:
        out["logVf"] = pd.to_numeric(df_raw[logVf_col], errors="coerce")
        if elogVf_col is not None:
            out["elogVf"] = pd.to_numeric(df_raw[elogVf_col], errors="coerce")
        elif eVf_col is not None:
            Vf = np.power(10.0, out["logVf"])
            eVf = pd.to_numeric(df_raw[eVf_col], errors="coerce")
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
        raise RuntimeError("Could not find Vf or log(Vf) columns in the .mrt file.")

    return out


def weighted_fit_free_slope(x: np.ndarray, y: np.ndarray, sy: np.ndarray) -> Tuple[float, float]:
    # weights in y
    if np.isfinite(sy).any():
        w = 1.0 / np.where(np.isfinite(sy) & (sy > 0), sy**2, np.nan)
        finite_w = w[np.isfinite(w)]
        if finite_w.size > 0:
            w = np.where(np.isfinite(w), w, np.nanmedian(finite_w))
        else:
            w = np.ones_like(y)
    else:
        w = np.ones_like(y)

    W = np.sum(w)
    xw = np.sum(w * x) / W
    yw = np.sum(w * y) / W
    Sxx = np.sum(w * (x - xw) ** 2)
    Sxy = np.sum(w * (x - xw) * (y - yw))
    b = Sxy / Sxx
    a = yw - b * xw
    return float(a), float(b)


def fixed_slope_a0_best(x_logV: np.ndarray, y_logM: np.ndarray, slope: float = 4.0) -> Tuple[float, float]:
    """
    For slope fixed (default 4), find best intercept c in:
        y = c + slope*x
    then convert c to a0 using:
        Mb = Vf^4/(G a0)
    where:
        y=log10(Mb/Msun), x=log10(Vf/km/s).
    """
    x = np.asarray(x_logV, dtype=float)
    y = np.asarray(y_logM, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size < 10:
        return (float("nan"), float("nan"))

    c = float(np.mean(y - slope * x))
    # Convert c -> a0.
    # Start from: Mb[Msun] = 10^y
    # Vf[km/s] = 10^x
    # a0 = (Vf[m/s])^4 / (G Mb[kg])
    #    = ( (10^x * 1000)^4 ) / (G * (10^y * Msun))
    # Taking log10:
    # log10(a0) = 4*(x + log10(1000)) - log10(G) - (y + log10(Msun))
    # But with y = c + 4x -> cancels x:
    # log10(a0) = 4*log10(1000) - log10(G) - (c + log10(Msun))
    log10_a0 = 4.0 * np.log10(KMS_TO_MS) - np.log10(G_SI) - (c + np.log10(MSUN_KG))
    a0 = 10.0 ** log10_a0
    # RMS around fixed slope
    yhat = c + slope * x
    rms = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return float(a0), rms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrt", required=True, help="Path to SPARC BTFR .mrt table (e.g., BTFR_Lelli2019.mrt)")
    ap.add_argument("--outdir", default="Enchan_BTFR_Test_Report", help="Output directory")
    ap.add_argument("--max_elogMb", type=float, default=None, help="Optional cut: keep rows with elogMb <= value")
    args = ap.parse_args()

    mrt_path = Path(args.mrt)
    if not mrt_path.exists():
        raise FileNotFoundError(f"Missing input file: {mrt_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = parse_mrt_fixedwidth(mrt_path)
    df = extract_btfr(df_raw)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["logMb", "logVf"]).reset_index(drop=True)

    if args.max_elogMb is not None and "elogMb" in df.columns:
        df = df.loc[df["elogMb"].isna() | (df["elogMb"] <= args.max_elogMb)].reset_index(drop=True)

    # Free-slope fit (benchmark)
    x = df["logVf"].to_numpy(dtype=float)
    y = df["logMb"].to_numpy(dtype=float)
    sy = df["elogMb"].to_numpy(dtype=float) if "elogMb" in df.columns else np.full_like(y, np.nan)

    a_fit, b_fit = weighted_fit_free_slope(x, y, sy)
    yhat = a_fit + b_fit * x
    resid = y - yhat
    rms_fit = float(np.sqrt(np.mean(resid**2)))

    df["yhat_free"] = yhat
    df["resid_free"] = resid

    # a0 implied per galaxy (Enchan/BTFR deep relation)
    Mb_msun = np.power(10.0, df["logMb"].to_numpy(dtype=float))
    Vf_kms = np.power(10.0, df["logVf"].to_numpy(dtype=float))
    a0_i = a0_from_btfr(Vf_kms, Mb_msun)
    df["a0_btfr_SI"] = a0_i
    df["log10_a0"] = np.log10(a0_i)

    p16, p50, p84 = robust_percentiles(a0_i, (16.0, 50.0, 84.0))
    lp16, lp50, lp84 = robust_percentiles(np.log10(a0_i), (16.0, 50.0, 84.0))

    # Fixed-slope (b=4) a0 estimate + RMS
    a0_fixed, rms_fixed = fixed_slope_a0_best(df["logVf"].to_numpy(), df["logMb"].to_numpy(), slope=4.0)

    # Save processed points
    df.to_csv(outdir / "btfr_points_processed.csv", index=False)

    # Summary CSVs
    summary_fit = pd.DataFrame([{
        "mrt_file": mrt_path.name,
        "sha256": sha256_file(mrt_path),
        "N": int(len(df)),
        "a_free": a_fit,
        "b_free": b_fit,
        "rms_dex_logMb_free": rms_fit,
    }])
    summary_fit.to_csv(outdir / "btfr_fit_summary.csv", index=False)

    summary_a0 = pd.DataFrame([{
        "mrt_file": mrt_path.name,
        "sha256": sha256_file(mrt_path),
        "N": int(len(df)),
        "a0_median_SI": p50,
        "a0_p16_SI": p16,
        "a0_p84_SI": p84,
        "log10_a0_median": lp50,
        "log10_a0_p16": lp16,
        "log10_a0_p84": lp84,
        "a0_fixed_slope4_SI": a0_fixed,
        "rms_dex_logMb_slope4": rms_fixed,
        "btfr_relation_used": "Mb = Vf^4 / (G a0)",
        "note": "a0_i computed per galaxy from (Vf^4)/(G Mb)",
    }])
    summary_a0.to_csv(outdir / "btfr_a0_summary.csv", index=False)

    # Figures
    plt.figure()
    plt.scatter(x, y, s=18, alpha=0.7)
    xs = np.linspace(np.nanmin(x) - 0.05, np.nanmax(x) + 0.05, 200)
    plt.plot(xs, a_fit + b_fit * xs)
    plt.xlabel("log10(Vf [km/s])")
    plt.ylabel("log10(Mb [Msun])")
    plt.title("SPARC BTFR (one point per galaxy)")
    plt.tight_layout()
    plt.savefig(outdir / "fig_btfr_points.png", dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(x, resid, s=18, alpha=0.7)
    plt.axhline(0.0)
    plt.xlabel("log10(Vf [km/s])")
    plt.ylabel("Residual in log10(Mb) [dex]")
    plt.title("BTFR residuals around best-fit line")
    plt.tight_layout()
    plt.savefig(outdir / "fig_btfr_residuals.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(np.log10(a0_i[np.isfinite(a0_i)]), bins=20)
    plt.xlabel("log10(a0 [m/s^2])")
    plt.ylabel("Count")
    plt.title("a0 implied by BTFR: a0 = Vf^4 / (G Mb)")
    plt.tight_layout()
    plt.savefig(outdir / "fig_a0_hist.png", dpi=200)
    plt.close()

    # Console summary
    print("BTFR reproduce summary")
    print(f"  mrt: {mrt_path.name}")
    print(f"  sha256: {sha256_file(mrt_path)}")
    print(f"  galaxies (N): {int(len(df))}")
    print(f"  fit (free): a={a_fit:.3f}, b={b_fit:.3f}, RMS={rms_fit:.3f} dex")
    print(f"  a0(BTFR): median={p50:.3e} m/s^2 (16–84%: {p16:.3e}–{p84:.3e})")
    print(f"           log10 a0 median={lp50:.3f} (16–84%: {lp16:.3f}–{lp84:.3f})")
    print(f"  a0(slope=4): {a0_fixed:.3e} m/s^2, RMS(slope=4)={rms_fixed:.3f} dex")
    print(f"Outputs written to: {outdir.resolve()}")
    print("Key files:")
    print("  - btfr_points_processed.csv")
    print("  - btfr_fit_summary.csv")
    print("  - btfr_a0_summary.csv")
    print("  - fig_btfr_points.png, fig_btfr_residuals.png, fig_a0_hist.png")


if __name__ == "__main__":
    main()