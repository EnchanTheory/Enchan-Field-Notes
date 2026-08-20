# Enchan Field Notes (Theoretical Framework)

**Current version:** v0.4.6
**Status:** theoretical note + public-data benchmark package
**License:** Enchan Research & Verification License v1.0 (see `LICENSE`)

Enchan Field Notes is a set of theoretical memoranda exploring whether several galaxy-scale regularities can be organized within an effective field description built around a dimensionless scalar `S`.

This repository includes:

* a LaTeX/PDF note (`main.pdf`, `sections/`)
* a small reproducibility package for public-data benchmarks (`enchan-theory-verification/`)

## Current theoretical synthesis

The v1.0.0 formulation of the Enchan Field is maintained in:

**The Enchan Field: An Effective Field Framework for Geometric Stabilization and Non-Linear Relaxation**  
https://github.com/EnchanTheory/The-Enchan-Field-Paper

This repository should be read as an earlier field-note and public-data benchmark precursor to that formulation. The later paper generalizes the original scalar-field notes into a finite-tension geometric stabilization framework and connects the continuous field formulation to deterministic relaxation on discrete graph topologies.

This work is not peer-reviewed. It does not claim to falsify particle dark matter.

---

## Scope (v0.4.6)

v0.4.6 is a **copyedited, minimal-scope release** intended for external distribution.

* The PDF focuses on definitions, effective equations, and galaxy-scale benchmark targets.
* Implementation details, device concepts, project planning, and forward roadmaps are intentionally out of scope.

---

## Benchmarks (public-data targets)

The reproducibility package defines and evaluates externally checkable targets commonly used in the literature:

1. **RAR / MDAR** (multi-point relation)
2. **BTFR** (one galaxy = one point)
3. **Rotation-curve shape stress test** (fixed-rule prediction, no per-galaxy tuning)

These are used as regression targets. They are not, by themselves, evidence of a unique underlying theory.

---

## Baseline closure (benchmark interface)

Some scripts use a compact empirical mapping as a benchmark interface (baseline-only):

[
g_{\mathrm{tot}} = \sqrt{g_{\mathrm{bar}}^2 + a_0,g_{\mathrm{bar}}}
]

This should be read as **reproducibility of the benchmark interface**, not as a completed derivation from a unique fundamental model.

---

## What's new in v0.4.6

* **Editorial / scope tightening:** removed nonessential narrative and non-public operational details from the TeX/PDF.
* **Consistency pass:** unified notation and kept the field description in a minimal effective form.
* **Benchmarks kept explicit:** benchmark definitions remain as externally checkable targets for later forward modeling.

---

## Repository structure

* `main.pdf`: compiled note (Enchan Field Notes)
* `sections/`: LaTeX sources
* `enchan-theory-verification/`: reproducibility package (scripts, outputs)

---

## Verification

Reproducibility instructions are provided inside:

* `enchan-theory-verification/README.md`

(Upstream public datasets are not committed to this repository.)

---

## Citation

If you wish to cite this work:

> Kobayashi, M. (2026). *Enchan Field Notes: Theoretical Framework*. v0.4.6. GitHub repository.

For the current v1.0.0 theoretical synthesis, see:

> Kobayashi, M. (2026). *The Enchan Field: An Effective Field Framework for Geometric Stabilization and Non-Linear Relaxation*. GitHub repository. https://github.com/EnchanTheory/The-Enchan-Field-Paper

---

## License

This project is licensed under the Enchan Research & Verification License v1.0.
See `LICENSE` for the full terms.

---

## Contact

For research collaboration, commercial licensing inquiries, or any questions regarding the Enchan Theory, please reach out to: `enchan.theory@gmail.com`

---

*Note: This is a living document and subject to change without notice.*

---