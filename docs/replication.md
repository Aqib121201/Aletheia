# Project Aletheia - Replication Guide

## Overview

This document provides comprehensive instructions for replicating all experiments, proofs, and simulations in Project Aletheia's α-fair, cryptographically-auditable allocation framework. Follow these steps to reproduce results and verify the theorem-first approach uniting ergodic control, convex geometry, stochastic processes, and zero-knowledge proofs.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows WSL2
- **Memory**: Minimum 8GB RAM, recommended 16GB+ for ZK proof generation
- **Storage**: 5GB free disk space for dependencies and data
- **Network**: Stable internet connection for dependency downloads

### Required Software

```
# Core dependencies
- Python 3.9+
- Rust 1.70+
- Lean 4.0+
- Git 2.30+
- Make 4.3+

# Optional but recommended
- Docker 20.10+ (for containerized reproduction)
- Node.js 18+ (for web dashboard)
```

## Installation Guide

### Step 1: Clone Repository

```
git clone https://github.com/samansiddiqui55/Aletheia.git
cd Aletheia
```

### Step 2: Environment Setup

#### Option A: Automatic Setup (Recommended)

```
# Creates virtual environment and installs all dependencies
make env
```

This command will:
- Create Python virtual environment in `venv/`
- Install Python dependencies from `pyproject.toml`
- Install Rust toolchain and dependencies
- Setup Lean 4 theorem prover
- Download sample datasets
- Configure ZK proof system

#### Option B: Manual Setup

```
# Python environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .

# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup install stable
rustup default stable

# Lean 4 installation
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source ~/.elan/env
elan install leanprover/lean4:stable
elan default leanprover/lean4:stable

# Additional dependencies
pip install numpy scipy matplotlib pandas
pip install cryptography pycrypto
pip install pytest pytest-cov black flake8
cargo install --git https://github.com/privacy-scaling-explorations/halo2.git
```

### Step 3: Verification

```
# Verify installation
make verify-setup

# Expected output:
# ✓ Python 3.9+ installed
# ✓ Virtual environment activated
# ✓ Rust toolchain ready
# ✓ Lean 4 theorem prover available
# ✓ ZK proof system configured
# ✓ All dependencies satisfied
```

## Core Replication Commands

### Full Replication Pipeline

```
# Run complete replication (takes ~15-30 minutes)
make reproduce
```

This executes the entire pipeline:
1. Data generation and preprocessing
2. Lean proof verification
3. Python simulation runs
4. ZK proof generation
5. Results validation
6. Report generation

### Individual Components

#### 1. Data Generation

```
# Generate Berlin sample data
python python/alethia/data_generator.py --city berlin --output data/sample/berlin_sample.json

# Generate synthetic allocation scenarios
python python/alethia/scenario_generator.py --scenarios 100 --output data/scenarios/

# Expected files:
# - data/sample/berlin_sample.json (19KB, 674 lines)
# - data/scenarios/scenario_001.json through scenario_100.json
```

#### 2. Lean Proof Verification

```
# Verify ergodicity theorems
cd proofs/
lean --make src/ergodicity.lean

# Expected output:
# Compiling src/ergodicity.lean
# ✓ theorem fairness_convergence
# ✓ theorem allocation_ergodic
# ✓ theorem temporal_consistency
# ✓ lemma drift_bounded
# All proofs verified successfully
```

#### 3. Python Simulation

```
# Run allocation simulations
python python/alethia/simulator.py --config configs/berlin_config.json --runs 1000

# Run with specific parameters
python python/alethia/simulator.py \
  --algorithm weighted_ergodic_fair \
  --fairness-constraint 0.95 \
  --efficiency-weight 0.8 \
  --output experiments/results/

# Expected output files:
# - experiments/results/allocation_results.json
# - experiments/results/fairness_metrics.json
# - experiments/results/efficiency_analysis.json
```

#### 4. Zero-Knowledge Proof Generation

```
# Generate sample proofs
python python/alethia/zk_interface.py --generate-proof --input data/sample/berlin_sample.json

# Batch proof generation
./zk/bench.sh --batch-size 10 --output zk/sample_proofs/

# Expected output:
# Generated proof: zk/sample_proofs/sample.proof (2.4KB)
# Verification time: 180ms
# Proof valid: ✓
```

## Detailed Experiment Reproduction

### Experiment 1: Basic Allocation Fairness

```
# Setup experiment
mkdir -p experiments/exp1_fairness
cd experiments/exp1_fairness

# Generate test data
python ../../python/alethia/simulator.py \
  --config ../../configs/fairness_test.json \
  --seed 42 \
  --output fairness_results.json

# Analyze results
python ../../python/alethia/analysis.py \
  --input fairness_results.json \
  --metrics gini_coefficient theil_index \
  --output fairness_analysis.pdf

# Expected metrics:
# Gini coefficient: 0.28 ± 0.02
# Theil index: 0.15 ± 0.01
# Fairness violations: < 5%
```

### Experiment 2: Cryptographic Auditability

```
# Setup ZK proof experiment
mkdir -p experiments/exp2_zk_proofs
cd experiments/exp2_zk_proofs

# Generate allocation with proof
python ../../python/alethia/zk_interface.py \
  --input ../../data/sample/berlin_sample.json \
  --algorithm groth16 \
  --output allocation_with_proof.json

# Verify proof independently
python ../../python/alethia/proof_verifier.py \
  --proof allocation_with_proof.json \
  --public-inputs public_params.json

# Expected verification output:
# Proof verification: PASSED
# Public inputs valid: ✓
# Zero-knowledge property: ✓
# Soundness check: ✓
```

### Experiment 3: Ergodic Convergence Analysis

```
# Long-term convergence test
mkdir -p experiments/exp3_convergence
cd experiments/exp3_convergence

# Run extended simulation
python ../../python/alethia/simulator.py \
  --config ../../configs/convergence_config.json \
  --timesteps 10000 \
  --track-convergence \
  --output convergence_data.json

# Analyze convergence properties
python ../../python/alethia/convergence_analyzer.py \
  --input convergence_data.json \
  --plot-trajectory \
  --output convergence_analysis.pdf

# Expected convergence metrics:
# Convergence time: < 500 timesteps
# Final deviation: < 0.001
# Ergodic property satisfied: ✓
```

### Experiment 4: Multi-District Allocation

```
# Berlin multi-district simulation
mkdir -p experiments/exp4_multi_district
cd experiments/exp4_multi_district

# Run district-wise allocation
for district in mitte kreuzberg pankow charlottenburg spandau; do
  python ../../python/alethia/simulator.py \
    --config ../../configs/district_configs/${district}.json \
    --output ${district}_results.json
done

# Aggregate and compare results
python ../../python/alethia/district_analyzer.py \
  --inputs *_results.json \
  --compare-fairness \
  --output multi_district_analysis.json

# Expected district comparison:
# Fairness variance across districts: < 0.05
# Resource utilization: > 90% all districts
# Cross-district equity: satisfied
```

## Performance Benchmarking

### Computational Performance

```
# Benchmark allocation algorithms
python python/alethia/benchmarks.py --suite allocation_performance

# Expected performance metrics:
# Algorithm: weighted_ergodic_fair
# - 1000 units, 5000 applicants: ~2.3s
# - Memory usage: < 512MB
# - Convergence iterations: < 100

# Algorithm: priority_weighted_fair
# - 1000 units, 5000 applicants: ~1.8s
# - Memory usage: < 384MB
# - Convergence iterations: < 75
```

### ZK Proof Performance

```
# Benchmark proof generation and verification
./zk/bench.sh --performance-test

# Expected ZK performance:
# Proof generation:
# - 100 allocations: 1.2s
# - 1000 allocations: 8.7s
# - 10000 allocations: 98.4s
# Proof verification:
# - Single proof: 180ms
# - Batch (10 proofs): 1.1s
# - Batch (100 proofs): 8.9s
```

## Data Validation

### Input Data Validation

```
# Validate Berlin sample data
python python/alethia/validators.py --validate-data data/sample/berlin_sample.json

# Expected validation output:
# ✓ Schema validation passed
# ✓ District data consistency
# ✓ Housing unit integrity
# ✓ Healthcare facility data valid
# ✓ Demographic group totals: 100%
# ✓ ZK proof hashes valid
```

### Output Validation

```
# Validate simulation results
python python/alethia/validators.py --validate-results experiments/results/sample_run.json

# Expected validation checks:
# ✓ Fairness constraints satisfied
# ✓ Resource conservation maintained
# ✓ Allocation totals consistent
# ✓ Temporal consistency verified
# ✓ Proof verification passed
```

## Troubleshooting Guide

### Common Issues

#### 1. Environment Setup Problems

```
# Issue: Python virtual environment not activating
# Solution:
rm -rf venv/
python3 -m venv venv
source venv/bin/activate
make env

# Issue: Rust compilation errors
# Solution:
rustup update stable
cargo clean
make clean-build
```

#### 2. Lean Proof Verification Failures

```
# Issue: Lean theorem compilation errors
# Solution:
cd proofs/
rm -rf build/
lake clean
lake exe cache get
lean --make src/ergodicity.lean

# Issue: Missing Lean dependencies
# Solution:
elan update
lake update
```

#### 3. ZK Proof Generation Issues

```
# Issue: Halo2 circuit compilation failure
# Solution:
cargo update
export RUST_LOG=debug
python python/alethia/zk_interface.py --debug --generate-proof

# Issue: Proof verification timeout
# Solution:
# Increase timeout in configs/zk_config.json
{
  "verification_timeout_ms": 5000,
  "batch_size": 5
}
```

#### 4. Memory and Performance Issues

```
# Issue: Out of memory during large simulations
# Solution:
# Reduce batch size in simulation config
python python/alethia/simulator.py --batch-size 100 --parallel-workers 2

# Issue: Slow convergence
# Solution:
# Adjust algorithm parameters
python python/alethia/simulator.py \
  --convergence-tolerance 0.01 \
  --max-iterations 500
```

## Expected Outputs and Results

### File Structure After Replication

```
Aletheia/
├── data/
│   ├── sample/berlin_sample.json          # 674 lines, 19KB
│   └── scenarios/scenario_*.json          # 100 scenario files
├── experiments/
│   └── results/
│       ├── sample_run.json                # Main simulation results
│       ├── fairness_metrics.json          # Fairness analysis
│       └── performance_report.pdf         # Performance benchmarks
├── zk/
│   └── sample_proofs/
│       ├── sample.proof                   # ZK proof file
│       └── verification_log.txt           # Verification results
└── proofs/
    └── build/                             # Compiled Lean proofs
```

### Key Result Files

#### sample_run.json Structure

```
{
  "metadata": {
    "experiment_id": "EXP_001",
    "timestamp": "2025-09-22T18:01:00+05:30",
    "algorithm": "weighted_ergodic_fair",
    "total_allocations": 1000
  },
  "results": {
    "fairness_metrics": {
      "gini_coefficient": 0.28,
      "theil_index": 0.15,
      "coefficient_of_variation": 0.22
    },
    "efficiency_metrics": {
      "resource_utilization": 0.94,
      "allocation_success_rate": 0.87,
      "average_processing_time": 2.3
    },
    "zk_verification": {
      "proofs_generated": 1000,
      "verification_success_rate": 1.0,
      "average_proof_size_kb": 2.4
    }
  }
}
```

### Performance Benchmarks

| Component | Expected Performance | Tolerance |
|-----------|---------------------|-----------|
| Data Loading | < 500ms | ±100ms |
| Allocation Algorithm | < 3s per 1000 units | ±0.5s |
| Proof Generation | < 200ms per proof | ±50ms |
| Proof Verification | < 200ms per proof | ±30ms |
| Full Pipeline | < 30min | ±5min |

### Quality Metrics

| Metric | Expected Range | Passing Threshold |
|--------|---------------|------------------|
| Gini Coefficient | 0.25 - 0.32 | < 0.35 |
| Theil Index | 0.12 - 0.18 | < 0.25 |
| Resource Utilization | > 90% | > 85% |
| User Satisfaction | > 4.0/5.0 | > 3.5/5.0 |
| Proof Verification | 100% | > 99% |

## Automated Testing

### Continuous Integration

```
# Run full test suite
make test

# Run specific test categories
make test-allocation      # Allocation algorithm tests
make test-fairness       # Fairness metric tests
make test-zk            # Zero-knowledge proof tests
make test-integration   # End-to-end integration tests

# Generate coverage report
make coverage
```

### Test Configuration

Create `.github/workflows/ci.yml` for automated testing:

```
name: Aletheia CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Environment
        run: make env
      - name: Run Tests
        run: make test
      - name: Run Replication
        run: make reproduce
```

## Support and Contact

For replication issues or questions:

- **GitHub Issues**: https://github.com/samansiddiqui55/Aletheia/issues
- **Documentation**: Check `docs/` directory for additional guides
- **Email**: Contact maintainers for specific technical issues

## Version History

- **v1.0.0**: Initial replication guide
- **v1.1.0**: Added ZK proof benchmarking
- **v1.2.0**: Enhanced troubleshooting section
- **v1.2.3**: Current version with full automation support

---

**Last Updated**: September 22, 2025  
**Tested Environments**: Ubuntu 20.04, macOS 12+, Windows WSL2  
**Estimated Replication Time**: 15-30 minutes (full pipeline)
```