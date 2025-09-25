"""
Project Aletheia - Fair Allocation Simulator

This module provides the experimental and benchmarking simulation framework for
evaluating α-fair, cryptographically-auditable allocation algorithms in housing,
healthcare, and synthetic environments. It supports reproducible research, batch runs,
parameter sweeps, result aggregation, and protocol-level audit integration.

Classes:
    AllocationSimulator: Runner for allocation experiments and benchmarks

Example Usage:
    >>> sim = AllocationSimulator(
    ...     dataset_path="data/sample/berlin_sample.json",
    ...     algorithms=["alpha_fairness", "entropy_duality"],
    ...     runs=100,
    ...     output_dir="experiments/results/"
    ... )
    >>> report = sim.run()
    >>> sim.save_results(report)
"""

import os
import json
import time
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from .allocator import create_allocator, AllocationResult

class AllocationSimulator:
    """
    Simulation and benchmarking framework for allocation algorithms.

    Parameters:
    ----------
    dataset_path : str
        JSON path to input dataset.
    algorithms : List[str]
        List of algorithm names to run (e.g. "alpha_fairness").
    runs : int
        Number of experimental runs per algorithm/scenario.
    batch_size : int
        How many runs to parallelize per batch (default=1).
    output_dir : str
        Directory for storing experiment results.
    random_seed : Optional[int]
        Base seed for reproducibility.
    verbose : bool
        Enable detailed logging.
    """

    def __init__(
        self,
        dataset_path: str,
        algorithms: List[str] = ["alpha_fairness"],
        runs: int = 1,
        batch_size: int = 1,
        output_dir: str = "experiments/results/",
        random_seed: Optional[int] = None,
        verbose: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        self.algorithms = algorithms
        self.runs = runs
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        self.verbose = verbose

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("AllocationSimulator")
        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def load_dataset(self) -> Dict[str, Any]:
        """Load dataset from JSON file."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def run(self) -> Dict[str, Any]:
        """Run all configured allocation algorithms and collect benchmarking metrics."""
        dataset = self.load_dataset()
        all_results = {}
        start_time = time.time()

        for algo in self.algorithms:
            algo_results = []
            self.logger.info(f"Running algorithm: {algo} ({self.runs} runs)")

            for r in range(self.runs):
                seed = (self.random_seed or 42) + r
                np.random.seed(seed)

                # Algorithm instantiation (supports kwargs pattern at call site)
                allocator = create_allocator(fairness_type=algo, random_seed=seed, verbose=self.verbose)
                try:
                    result: AllocationResult = allocator.allocate(dataset)
                    algo_results.append(result)
                    self.logger.debug(f"Run {r+1} of {algo} completed: Gini = {result.fairness_metrics.get('gini_coefficient'):.3f}")
                except Exception as e:
                    self.logger.warning(f"Run {r+1} of {algo} failed: {e}")

            all_results[algo] = {
                "runs": len(algo_results),
                "metrics": self.aggregate_metrics([r.fairness_metrics for r in algo_results]),
                "convergence": self.aggregate_convergence([r.convergence_info for r in algo_results]),
                "allocations": [r.allocations for r in algo_results],
                "experiment_metadata": self.make_metadata(algo, algo_results)
            }

        elapsed = time.time() - start_time
        all_results["simulation_time_seconds"] = elapsed
        self.logger.info(f"All simulations completed in {elapsed:.2f} seconds")
        return all_results

    def aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """Aggregate fairness and efficiency metrics over multiple runs."""
        if not metrics_list:
            return {}

        keys = list(metrics_list[0].keys())
        agg = {}
        for key in keys:
            values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (float, int))]
            if values:
                agg[key + "_mean"] = float(np.mean(values))
                agg[key + "_std"] = float(np.std(values))
                agg[key + "_min"] = float(np.min(values))
                agg[key + "_max"] = float(np.max(values))
        return agg

    def aggregate_convergence(self, conv_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate convergence information (iterations, status, gradient norm, etc)."""
        n = len(conv_list)
        if n == 0:
            return {}
        iterations = [conv.get("iterations", 0) for conv in conv_list]
        statuses = [conv.get("status", "unknown") for conv in conv_list]
        all_statuses = {s: statuses.count(s) for s in set(statuses)}
        return {
            "iterations_mean": np.mean(iterations),
            "iterations_std": np.std(iterations),
            "convergence_statuses": all_statuses
        }

    def make_metadata(self, algorithm: str, run_results: List[AllocationResult]) -> Dict[str, Any]:
        """Compose metadata for experiment reproducibility and audit."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        agent_count = run_results[0].allocation_matrix.shape[0] if run_results else 0
        resource_count = run_results[0].allocation_matrix.shape[1] if run_results else 0
        return {
            "algorithm": algorithm,
            "dataset_file": str(self.dataset_path),
            "runs": len(run_results),
            "timestamp": timestamp,
            "agent_count": agent_count,
            "resource_count": resource_count,
            "code_version": "1.2.3",
            "random_seed": self.random_seed,
        }

    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save simulation results to output directory as a JSON file."""
        if not filename:
            filename = f"simulation_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as out:
            json.dump(results, out, indent=2)
        self.logger.info(f"Results written to: {path}")

    def benchmark(self, full: bool = False):
        """Run benchmarking on all supported algorithms/datasets."""
        import pprint
        self.logger.info("Starting benchmarking run")
        report = self.run()
        self.logger.info("Benchmarking run completed")
        pprint.pprint(report)
        self.save_results(report)

# CLI support
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Project Aletheia - Allocation Simulation Runner")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input dataset JSON")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["alpha_fairness"], help="Algorithms to test")
    parser.add_argument("--runs", type=int, default=1, help="Runs per algorithm")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for parallel execution")
    parser.add_argument("--output-dir", type=str, default="experiments/results/", help="Output directory")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for experiment")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    sim = AllocationSimulator(
        dataset_path=args.dataset,
        algorithms=args.algorithms,
        runs=args.runs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        verbose=args.verbose,
    )
    results = sim.run()
    sim.save_results(results)
