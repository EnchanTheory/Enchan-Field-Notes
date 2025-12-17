# Enchan Field Notes (Theoretical Framework)

**Current version:** v0.4.0  
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

## Benchmark closure (used as a reproducibility baseline)

Several scripts use an effective one-parameter mapping as a working hypothesis:

$$
g_{\mathrm{tot}} = \sqrt{g_{\mathrm{bar}}^2 + a_0\,g_{\mathrm{bar}}}
$$

This repository reproduces the above benchmarks using standard SPARC-style definitions and
reports scatter/diagnostics. This should be read as **reproducibility of the benchmark mapping**,
not as a completed derivation from a unique fundamental theory.
In v0.4.0, Eq. 6.6.2/6.6.3 modify the effective acceleration scale used inside this benchmark-style closure for regime tests.

---

## What's new in v0.4.0

### 1) Eq. 6.6.2 is now explicit (Galaxy: Phi-screening / “pinning”)
We introduce an explicit **environmental suppression** term in deep baryonic potentials:
an effective acceleration scale
\( a_{0,\mathrm{eff}} = a_{0,\mathrm{free}} \, \mathcal{S}_\Phi(|\Phi_{\mathrm{bar}}|) \)
with a minimal pinning function parameterized by global \((\Phi_c, n)\).
This term is evaluated and constrained on galaxy data via cross-validation (C1 pinning test).

### 2) Eq. 6.6.3 is separated as an optional Solar-System safeguard (High-g suppression)
Solar-System extrapolations can be sensitive to absolute-acceleration residuals in high-acceleration regimes.
We therefore separate a **high-acceleration suppression** factor
\( \mathcal{S}_g(g_N) \)
as an optional safeguard used only for Solar-System diagnostic scripts.
It is **not used** in galaxy calibration / cross-validation runs.

### 3) Code separation for reproducibility
- `enchan_core_model_plus.py` implements **Eq. 6.6.2 only** (Phi-screening / pinning).
- `enchan_core_model_g_screening.py` implements **Eq. 6.6.3 only** (High-g suppression).
This separation is intentional to avoid mixing Solar-System safeguards into galaxy calibration.

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
  from the [SPARC website](https://astroweb.case.edu/SPARC/)
- Run the scripts as described in `enchan-theory-verification/README.md`
- For v0.4.0, see the `v0.4.0` section in `enchan-theory-verification/README.md` for the exact script set and commands.

---

## Disclaimer

This repository contains theoretical and cosmological descriptions.
Technological implementations, specific device architectures (e.g., "Enchan-001" referenced in the text),
and industrial applications are out of scope and are not disclosed here.

---

## Citation

If you wish to cite this work:

> Kobayashi, M. (2025). *Enchan Field Notes: The Inception World and Topological Defects*. v0.4.0. GitHub Repository.

---

## License

This repository uses a hybrid licensing scheme:

- **Source code / verification scripts (`enchan-theory-verification/`):**
  MIT License (see `LICENSE`)
- **Textual content of Enchan Field Notes (`main.tex`, PDF, and `sections/`):**
  Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

You are free to share and adapt the text and PDF for **non-commercial** purposes,
as long as you provide proper attribution and indicate if changes were made.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17966704.svg)](https://doi.org/10.5281/zenodo.17966704)

---

*Note: This is a living document and subject to change without notice.*
*Note: The Zenodo DOI may correspond to a tagged release (e.g., v0.3.3); this README describes the current GitHub state v0.4.0.*