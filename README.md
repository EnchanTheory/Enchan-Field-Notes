# Enchan Field Notes (Theoretical Framework)

**Current version:** v0.3.3  
**Status:** speculative research note + public-data reproducibility package

Enchan Field Notes is a collection of theoretical memoranda exploring whether several
well-known galaxy-scale regularities can be described by an effective field picture
(topological defects in a dimensionless "time-dilation" field `S`).

This repository includes:
- a LaTeX/PDF note (`main.pdf`, `sections/`)
- a small Python reproducibility package based on public SPARC products (`enchan-theory-verification/`)

This work is not peer-reviewed. It does not claim to falsify particle dark matter.
In v0.3.x, several scripts use an **effective closure** as a benchmark mapping (see below);
a field-theoretic derivation is a theory-side goal rather than an established result.

---

## What is reproduced here (public-data benchmarks)

The reproducibility package focuses on three SPARC-based empirical regularities:

1. **RAR / MDAR**: across many radii and galaxies, observed acceleration tightly tracks baryonic acceleration.
2. **BTFR**: baryonic mass tightly correlates with outer/flat rotation velocity.
3. **Rotation-curve shape stress test**: with baryonic components + a single fixed rule, one can generate nontrivial curve shapes across a large sample (no per-galaxy tuning).

These are treated as **externally checkable targets** and regression tests.

---

## The benchmark closure used in v0.3.x scripts

Several scripts use an effective one-parameter mapping as a working hypothesis:

$$
g_{\mathrm{tot}} = \sqrt{g_{\mathrm{bar}}^2 + a_0\,g_{\mathrm{bar}}}
$$

This repository reproduces the above benchmarks using standard SPARC-style definitions and
reports scatter/diagnostics. This should be read as **reproducibility of the benchmark mapping**,
not as a completed derivation from a unique fundamental theory.

---

## What's new in v0.3.3

- **Theory-side refinement (Chapter 6):** We reformulate the "surface-density anchor" as a **pressure-matching / stress-scale** condition, providing a compact dimensional motivation for a scaling of the form $V_0 \propto G \Sigma_b^2$ (up to order-unity factors and proxy systematics). This should be read as an EFT-level organizing principle, not yet a first-principles derivation.

- **Differential Prediction Test (Test C1):** We added a 5-fold cross-validation test comparing:
  (A) a fixed-$a_0$ baseline vs. (B) a proxy-driven variable-$a_0$ model.
  - **Global (this run):** performance is similar overall (e.g., Mean RMS differs by ~0.003 dex; Win Rate ~52%).
  - **Stratified (by SB_proxy quartiles):** the variable-$a_0$ model performs better in the **low-SB quartile** (e.g., Win Rate ~73% in Q1), while degrading in the highest-SB quartile. This pattern is consistent with (i) a cleaner disk proxy regime at low SB and/or (ii) proxy/structure limitations in dense inner regions.

- **Traceability upgrades:** the pipeline now exports reproducibility metadata (hashes, run config) and diagnostic lists (dropped-by-filter and name-collision logs) to support auditing and re-analysis.

---

## Repository structure

- `main.pdf`: compiled note (Enchan Field Notes)
- `sections/`: LaTeX sources
- `enchan-theory-verification/`: reproducibility package (scripts, outputs, hashes)

---

## Verification (quick start)

The reproducibility package has its own README with detailed instructions:

- Go to `enchan-theory-verification/`
- Download SPARC public products:
  - `Rotmod_LTG.zip`
  - `BTFR_Lelli2019.mrt`
  from the [SPARC website](http://astroweb.case.edu/SPARC/)
- Run the scripts as described in `enchan-theory-verification/README.md`

---

## Disclaimer

This repository contains theoretical and cosmological descriptions.
Technological implementations, specific device architectures (e.g., "Enchan-001" referenced in the text),
and industrial applications are out of scope and are not disclosed here.

---

## Citation

If you wish to cite this work:

> Kobayashi, M. (2025). *Enchan Field Notes: The Inception World and Topological Defects*. v0.3.3. GitHub Repository.

---

## License

This repository uses a hybrid licensing scheme:

- **Source code / verification scripts (`enchan-theory-verification/`):**
  MIT License (see `LICENSE`)
- **Textual content of Enchan Field Notes (`main.tex`, PDF, and `sections/`):**
  Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

You are free to share and adapt the text and PDF for **non-commercial** purposes,
as long as you provide proper attribution and indicate if changes were made.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17953210.svg)](https://doi.org/10.5281/zenodo.17953210)

---

*Note: This is a living document and subject to change without notice.*