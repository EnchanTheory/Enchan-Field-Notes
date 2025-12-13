# VALIDATION (BTFR)

This checklist is the v0.2.4 gate for the BTFR asset.

## What to verify
1) The script runs end-to-end on a locally downloaded SPARC BTFR `.mrt` table.
2) The printed summary is recorded (including input sha256).
3) Output artifacts exist and are consistent with the printed summary.

## Run
```bash
python enchan_btfr_reproduce.py --mrt BTFR_Lelli2019.mrt
```

## Expected artifacts
- `Enchan_BTFR_Test_Report_v0_1/btfr_points_processed.csv`
- `Enchan_BTFR_Test_Report_v0_1/btfr_fit_summary.csv`
- `Enchan_BTFR_Test_Report_v0_1/fig_btfr_points.png`
- `Enchan_BTFR_Test_Report_v0_1/fig_btfr_residuals.png`
- `Enchan_BTFR_Test_Report_v0_1/enchan_btfr_test_report_v0p1.tex`

## Record here (paste your stdout)
- mrt filename:
- sha256:
- N:
- a, b, RMS:

## Acceptance
- Re-running with the same `.mrt` file yields the same sha256 and the same `btfr_fit_summary.csv` values (within rounding).
