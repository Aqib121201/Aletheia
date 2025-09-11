# Project Aletheia — α-fair, cryptographically-auditable allocation framework

[![CI](https://img.shields.io/badge/ci-passing-brightgreen)](./.github/workflows/repro.yml)
[![Proofs](https://img.shields.io/badge/proofs-lean--sketch-blue)](proofs/)
[![Reproducibility](https://img.shields.io/badge/reproducible-in_progress-orange)](docs/replication.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

**Status:**  *Research in progress — scaffold live, core proofs & ZK components under construction.*  
**One-line:** Building a theorem-first, cryptographically-auditable allocation system uniting ergodic control, convex geometry, stochastic processes, and zero-knowledge proofs (housing & healthcare).

---

## Quick signal (what to check)
- `proofs/` — Lean lemma scaffold + LaTeX theorem sketches.  
- `python/alethia/` — simulator scaffold, zk-interface.  
- `zk/sample_proofs/sample.proof` — placeholder proof (downloadable).  
- `docs/replication.md` — exact commands, expected outputs.

---

## Current status
-  Python simulator scaffold (seeded runs).  
-  Lean proof skeleton (`proofs/src/ergodicity.lean`) with a toy lemma.  
-  CI workflow runs tests + reproduce target.  
-  ZK circuits: shim + bench script; Halo2 circuits next.  
-  Fairness-drift dashboard (design ready).

---
