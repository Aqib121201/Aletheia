# Project Aletheia — α-fair, cryptographically-auditable allocation framework

[![CI](https://img.shields.io/badge/ci-passing-brightgreen)](./.github/workflows/repro.yml)
[![Reproducibility](https://img.shields.io/badge/reproducible-in_progress-orange)](docs/replication.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

**Status:**  *Research in progress — scaffold live, core proofs & ZK components under construction.*  
**One-line:** Building a theorem-first, cryptographically-auditable allocation system uniting **ergodic control, convex geometry, stochastic processes, and zero-knowledge proofs**, applied to **housing & healthcare allocation**.

---

##  Vision & Next Horizon
Project Aletheia aims to provide **provable fairness and cryptographic auditability** in online resource allocation.  
We combine **formal proofs (Lean/Coq)**, **optimization theory**, and **zero-knowledge verification** to ensure allocations are *fair, transparent, and EU AI-Act compliant*.

---

##  Objectives (2026 Roadmap)
- **Ergodic Allocation Law** → formalize almost-sure fairness for multi-resource scheduling (housing, hospital beds) under adversarial arrivals.  
- **Minimax Regret** → target O(√n log d) bounds with polylog-time mirror descent projections.  
- **Entropy–Fairness Duality** → prove Banach–Mazur distortion ≤ O(log d).  
- **ZK Allocation Checkpoints** → ≤50 ms verification, ≤2 MB proof size; enforce GDPR/AI-Act compliance.  
- **Berlin–Munich–Hamburg Simulator** → reproducible Rust/Python implementation + fairness-drift dashboard.  
- **Formal Proofs & Open Code** → Lean/Coq, dataset cards, CI, >80% coverage.

---

## Engineering Principles
- **Theorem-First Science** → prove guarantees before coding.  
- **Verifiable by Design** → every allocation emits a Halo2/PLONK ZK-certificate bound to logs & DP noise.  
- **Geometry-Aware Optimization** → Bregman divergences tuned to fairness polytopes.  
- **Stochastic-Control Guarantees** → Lyapunov drift ensures ε-bounded violations at scale.  
- **EU-Ready** → explicit DPIA & AI-Act templates.  
- **Reproducible & Open** → one-click CI, seeded runs, dataset cards.  

---

##  Repository Structure (current scaffold)
```

project-aletheia/
├── README.md
├── python/alethia/       # simulator + allocator stub
├── proofs/               # Lean lemma scaffolding
├── zk/                   # placeholder ZK circuits
├── experiments/results/  # seeded runs (synthetic data)
├── docs/                 # replication notes, AI-Act DPIA (draft)
└── .github/workflows/    # CI reproducibility checks

````

---

##  Current Status (Sep 2025)
- **Python simulator scaffold** (`python/alethia/`) — generates synthetic arrivals + proportional allocations.  
-  **Lean proof skeleton** (`proofs/ergodicity.lean`) — placeholder lemma formalized for toy model.  
-  **CI workflow** — runs simulator + tests on push.  
-  **ZK checkpoint prototype** — simple invariant checks, Halo2 circuits next.  
-  **Fairness-drift dashboard** — planned with Berlin–Munich–Hamburg datasets.  
-  **Entropy–Fairness duality theorem** — design & experiments ongoing.  

---

## 🛠️ Quickstart (scaffold)
```bash
git clone https://github.com/<you>/project-aletheia.git
cd project-aletheia
make env && make reproduce
````

Produces:

* `experiments/results/sample_run.json` (seeded allocations)
* `zk/sample_proofs/sample.proof` (placeholder proof artifact)

---

##  Early Prototype Metrics

| Component             | Current (scaffold) | Target (2026) |
| --------------------- | -----------------: | ------------: |
| Proof artifact size   |       22 KB (stub) |        ≤ 2 MB |
| Verifier time         |      \~8 ms (stub) |       ≤ 50 ms |
| Allocations simulated |      10⁴ synthetic |     10⁶+ real |
| Coverage (Python)     |                65% |         ≥ 80% |

---

##  Contributors

## 👥 Contributors

* **Aqib Siddiqui** — Lead researcher & developer: formal proofs (Lean), ergodic allocation theory, Rust/Python simulator scaffolding, ZK checkpoint prototype.
* *Open for collaboration — see* [`CONTRIBUTING.md`](CONTRIBUTING.md).
---


## 📅 Roadmap (next 6 months)

* Q4 2025 → extend Lean proofs, ZK checkpoint prototype.
* Q1 2026 → Berlin–Munich–Hamburg datasets, fairness-drift dashboard.
* Q2 2026 → entropy–fairness theorem + minimax regret proofs.

---

## 📜 License

Apache 2.0

---
