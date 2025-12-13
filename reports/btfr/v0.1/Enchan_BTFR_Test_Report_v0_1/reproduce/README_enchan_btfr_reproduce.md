# Enchan BTFR reproduce

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

## Outputs (written to `Enchan_BTFR_Test_Report_v0_1/` by default)
- `btfr_points_processed.csv`
- `btfr_fit_summary.csv`
- `fig_btfr_points.png`
- `fig_btfr_residuals.png`
- `enchan_btfr_test_report_v0p1.tex`

## Dependencies
- numpy
- pandas
- matplotlib

No SciPy required.
