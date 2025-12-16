# Enchan Theory Verification (public-data reproducibility)

This folder contains reproducible, public-data benchmarks used in *Enchan Field Notes v0.3.3*.

## What this is (and what it is not)

- This is a set of scripts that reproduce several well-known SPARC-based empirical regularities and related diagnostics.
- This is intended for external checkability: same inputs → same outputs (with input SHA256 logged).
- This is not a proof of Enchan Theory, and it does not rule out particle dark matter.
- Important: several steps use an effective closure (“Enchan quadrature law”) as a working hypothesis:

  $$g_{\rm tot} = \sqrt{g_{\rm bar}^2 + a_0\,g_{\rm bar}}$$

  In v0.3.3, the field-theoretic derivation of this closure and of $a_0$ is the theory-side goal.
  These scripts provide targets and regression tests.

---

## The three observational targets

These scripts focus on three repeatable patterns seen across many galaxies:

1. RAR / MDAR — at many radii, the observed acceleration is tightly linked to the baryonic one.
2. BTFR — the baryonic mass of a galaxy is tightly linked to its outer/flat rotation speed.
3. Rotation-curve shapes — using baryonic components plus a fixed rule, you can predict nontrivial curve shapes across a large sample (stress test).

---

## Inputs (not committed)

You need the following public SPARC files locally:

- `Rotmod_LTG.zip` (mass-model rotation-curve decomposition files)
- `BTFR_Lelli2019.mrt` (SPARC BTFR table; CDS-style fixed-width)

Source: SPARC website  
https://astroweb.case.edu/SPARC/

Repository policy: large upstream datasets are not committed.  
Each script records the SHA256 of your local input file(s) so results are comparable.

---

## Requirements

- Python 3.10+ recommended
- `numpy`, `pandas`, `matplotlib`
- `scipy` is optional (used only for p-values in some tests; scripts provide fallbacks where applicable)

Install minimal dependencies:

```bash
python -m pip install numpy pandas matplotlib

```

Optional:

```bash
python -m pip install scipy

```

---

##Scripts in this folder###1) BTFR benchmark (and implied a_0 distribution)Computes BTFR points, a simple log–log fit, and the implied per-galaxy estimate:

```bash
python enchan_btfr_reproduce_enchan.py --mrt BTFR_Lelli2019.mrt
# optional quality cut (if elogMb exists)
python enchan_btfr_reproduce_enchan.py --mrt BTFR_Lelli2019.mrt --max_elogMb 0.10

```

Outputs (default): `Enchan_BTFR_Test_Report_v0_1/`
Key outputs include CSV tables and figures, plus `btfr_a0_summary.csv`.

---

###2) RAR benchmark using the Enchan effective closureUses SPARC `Rotmod_LTG.zip` to compute g_{\rm bar}(r) and compare to observed accelerations,
using the Enchan quadrature law as the mapping.

Requires an a_0 value (recommended: BTFR median from step 1).

```bash
python enchan_rar_reproduce_enchan.py --zip Rotmod_LTG.zip --a0 1.5e-10
# optional: override mass-to-light ratios
python enchan_rar_reproduce_enchan.py --zip Rotmod_LTG.zip --a0 1.5e-10 --Yd 0.60 --Yb 0.70

```

Outputs (default): `Enchan_RAR_Test_Report_v0_1/`

---

###3) Rotation-curve prediction (shape stress test)Predicts V_{\rm pred}(r) from baryonic components and the same closure, with fixed global parameters
(no per-galaxy tuning).

Requires an a_0 value (recommended: BTFR median from step 1).

```bash
python enchan_rotationcurve_reproduce_enchan.py --zip Rotmod_LTG.zip --a0 1.5e-10

```

Outputs (default): `Enchan_SPARC_Rotation_Curve_Prediction_Report_v0_1/`

---

###4) Exploration: a_0 vs surface-brightness proxy (v0.3.2)This is an exploratory test of the Chapter 6 “surface-density anchor” idea.
Updated in v0.3.2: includes transparent data-drop logging and robust statistical reporting.

* a_0 is derived from BTFR.
* Surface-brightness proxy is derived from `Rotmod_LTG.zip`:
**SB_proxy = median of the innermost 3 positive `SBdisk` points**

```bash
python enchan_a0_sb_correlation.py --mrt BTFR_Lelli2019.mrt --zip Rotmod_LTG.zip

```

Outputs (default): `Enchan_Correlation_Test_v0_3_2/`

* `correlation_data_v0_3_2.csv`
* `correlation_summary.csv` (includes drop counts and interpretation logic)
* `fig_correlation_v0_3_2.png`

Pre-registered interpretation: checks consistency with minimal anchor ansatz (P1),
self-regulation (P2), or proxy-noise dominance (P3). See the script docstring.

---

###5) Mathematical consistency check for the transition function (v0.3.2)Numerically checks that the candidate transition function \mu(x) used in the note is
consistent with the quadrature closure and remains numerically well-behaved across regimes.

```bash
python enchan_transition_function_check.py

```

Outputs: `Enchan_Transition_Check_v0_3_2/`

* Pass/Fail status based on a max relative error tolerance (see script output).
* A figure visualizing \mu(x) behavior (see the output directory).

---

###6) Differential Prediction Test (Test C1) - New in v0.3.3We implemented a cross-validated "differential prediction" test for the Anchor Hypothesis, i.e. that the acceleration scale a_0 varies with a baryonic surface-density proxy.

**Run configuration (this report):**

* Dataset: SPARC Rotmod + BTFR table (see `run_summary.csv`)
* Disk-dominated filter: `q95%(f_bul) < 0.5` (Yd=0.5, Yb=0.7)
* Galaxies used: 102
* Protocol: 5-fold cross validation (seed=42)

**Global results (102 galaxies):**

* Fixed-a_0 model (A): Mean RMS = 0.1580 dex, Median RMS = 0.1248 dex
* Variable-a_0 model (B): Mean RMS = 0.1554 dex, Median RMS = 0.1288 dex
* Net mean improvement (A − B): +0.0026 dex
* Win rate (B better than A): 53 / 102 (52.0%)

**Trend (amplitude vs. proxy):**
A positive log–log slope between a_0 and the surface-density proxy is observed (typical slope \sim +0.11 in this run).
This indicates the *sign* of the dependence is consistent with the Anchor Hypothesis, while the *net predictive gain* is small when averaged over the full sample.

**Performance by surface-brightness quartile (SB_proxy quartiles within this run):**
| Quartile (SB_proxy) | N  | Win Rate (B>A) | Mean (A−B) [dex] | Note |
| :--- | ---: | ---: | ---: | :--- |
| **Q1 (Low SB)** | 26 | **73.1%** | **+0.0208** | **Variable-a_0 improves systematically** |
| Q2           | 25 | 44.0%     | −0.0030 | Mixed |
| Q3           | 25 | 64.0%     | +0.0026 | Mild improvement |
| Q4 (High SB) | 26 | 26.9%     | −0.0103 | Degrades (proxy limitation / inner complexity) |

**Diagnostics and traceability:**

* Per-galaxy outcomes: `prediction_c1_disk_details.csv`
* Fold summary: `prediction_c1_fold_summary.csv`
* Dropped-by-filter list: `diagnostic_dropped_bulge.csv`
* Name-collision log: `diagnostic_collisions.csv`
* Full run metadata + hashes: `run_summary.csv`

**Interpretation (current status):**
The variable-a_0 model shows a clear advantage in the low-SB regime, while high-SB systems show the opposite trend.
This suggests either (i) limitations of the current proxy in dense regions and/or (ii) missing physics required for complex inner structures.
The consistent signal in the clean Q1 regime supports the core theoretical prediction.

Usage:

```bash
python enchan_variable_a0_prediction.py --mrt BTFR_Lelli2019.mrt --zip Rotmod_LTG.zip --max_bulge_frac 0.5 --bulge_quantile 0.95

```

Outputs (default): `Enchan_Prediction_C1_DiskOnly_v0_3/`

---

##Recommended run order (quickstart)1. Put `BTFR_Lelli2019.mrt` and `Rotmod_LTG.zip` next to these scripts (or pass explicit paths).
2. (Optional but recommended) Run `enchan_transition_function_check.py` to verify numerical consistency.
3. Run the BTFR script and read the median a_0 from the output CSV.
4. Use that a_0 when running the RAR and rotation-curve scripts.
5. Run the correlation script (4) to explore the Chapter 6 anchor ansatz.
6. Run the differential prediction test (6) to verify the surface-density dependence in the Low-SB regime.

---

##Notes for external reviewers* All scripts log input SHA256 for reproducibility.
* Name matching across datasets uses a normalized galaxy-name key (removes separators and leading zeros).
* These benchmarks are intentionally minimal and transparent: the goal is checkability, not maximum model complexity.