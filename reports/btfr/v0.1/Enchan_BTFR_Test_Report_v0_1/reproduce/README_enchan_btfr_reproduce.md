# Enchan BTFR reproduce

> **Status (baseline / non-Enchan):**  
> **Not derived from Enchan equations. Baseline only.**  
> This folder reproduces a published SPARC BTFR table fit (one galaxy = one point) and computes the implied
> acceleration scale per galaxy, `a0 = Vf^4 / (G * Mb)`.  
> It is kept as a benchmark/regression test for future Enchan-derived predictions.

This folder contains a minimal, reproducible BTFR benchmark from a SPARC BTFR `.mrt` table.

## Input (not committed)
Place the SPARC BTFR table next to the script, for example:

- `BTFR_Lelli2019.mrt`

## Run
```bash
python enchan_btfr_reproduce.py --mrt BTFR_Lelli2019.mrt
```

Optional quality cut (if `elogMb` exists in the table):
```bash
python enchan_btfr_reproduce.py --mrt BTFR_Lelli2019.mrt --max_elogMb 0.10
```

Optional TeX stub:
```bash
python enchan_btfr_reproduce.py --mrt BTFR_Lelli2019.mrt --make-tex
```

## Outputs (written to `Enchan_BTFR_Test_Report_v0_1/` by default)
- `btfr_points_processed.csv` (includes per-galaxy `a0_m_s2` and `log10_a0_m_s2`)
- `btfr_fit_summary.csv` (a, b, RMS, sha256)
- `btfr_a0_summary.csv` (median and 16/84 percentiles of a0)
- `fig_btfr_points.png`
- `fig_btfr_residuals.png`
- `fig_a0_hist.png`
- `enchan_btfr_test_report_v0p1.tex` (only if `--make-tex`)

## Dependencies
- numpy
- pandas
- matplotlib

No SciPy required.
