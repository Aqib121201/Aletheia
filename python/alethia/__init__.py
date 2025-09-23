"""
Project Aletheia - α-fair, Cryptographically-Auditable Allocation Framework

A theorem-first approach to fair resource allocation uniting ergodic control,
convex geometry, stochastic processes, and zero-knowledge proofs.

Main components:
- Allocation algorithms with α-fairness guarantees
- Zero-knowledge proof generation and verification
- Formal verification interfaces with Lean 4
- Domain-specific applications (housing, healthcare)
- Comprehensive fairness metrics and analysis tools

Example usage:
    >>> import aletheia as ale
    >>> data = ale.load_dataset("berlin_housing")
    >>> allocator = ale.AlphaFairnessAllocator(alpha=1.0)
    >>> result = allocator.allocate(data)
    >>> proof = ale.generate_zk_proof(result)
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

# Version and metadata
__version__ = "1.2.3"
__author__ = "Saman Siddiqui, Aqib Siddiqui"
__email__ = "saman.siddiqui@aletheia-project.org"
__license__ = "MIT"
__copyright__ = "2025, Project Aletheia Contributors"

# Package information
__title__ = "aletheia"
__description__ = "α-fair, Cryptographically-Auditable Allocation Framework"
__url__ = "https://github.com/samansiddiqui55/Aletheia"
__status__ = "Development"

# Minimum Python version check
if sys.version_info < (3, 9):
    raise RuntimeError("Project Aletheia requires Python 3.9 or later")

# Core algorithm imports
try:
    from .algorithms.fairness import (
        AlphaFairnessAllocator,
        ProportionalFairnessAllocator,
        MaxMinFairnessAllocator,
        EntropyDualityAllocator,
        WeightedFairnessAllocator,
    )
    
    from .algorithms.mechanisms import (
        WeightedLotteryMechanism,
        PriorityBasedMechanism,
        AuctionMechanism,
        SerialDictatorshipMechanism,
    )
    
    from .algorithms.optimization import (
        ConvexOptimizer,
        ErgodicController,
        GradientProjectionSolver,
        InteriorPointSolver,
    )
    
except ImportError as e:
    warnings.warn(f"Could not import core algorithms: {e}")
    # Create placeholder classes to prevent import errors
    class AlphaFairnessAllocator:
        def __init__(self, *args, **kwargs):
            raise ImportError("Core algorithm modules not available")

# Zero-knowledge proof system
try:
    from .zk import (
        ZKProofSystem,
        Groth16Backend,
        Halo2Backend,
        PlonkBackend,
        generate_zk_proof,
        verify_zk_proof,
        batch_verify_proofs,
    )
    
    from .zk.circuits import (
        AllocationCircuit,
        FairnessCircuit,
        TemporalConsistencyCircuit,
    )
    
except ImportError as e:
    warnings.warn(f"Zero-knowledge components not available: {e}")
    
    def generate_zk_proof(*args, **kwargs):
        raise ImportError("ZK proof system not available. Install with: pip install aletheia[zk]")
    
    def verify_zk_proof(*args, **kwargs):
        raise ImportError("ZK proof system not available")

# Fairness metrics and analysis
try:
    from .metrics import (
        compute_gini_coefficient,
        compute_theil_index,
        compute_entropy_fairness_duality,
        compute_envy_ratio,
        compute_utilitarian_welfare,
        compute_egalitarian_welfare,
        FairnessAnalyzer,
        InequalityMeasures,
    )
    
except ImportError as e:
    warnings.warn(f"Metrics module not available: {e}")

# Data handling and validation
try:
    from .data import (
        Dataset,
        BerlinHousingDataset,
        HealthcareDataset,
        SyntheticDataset,
        load_dataset,
        validate_dataset,
        generate_synthetic_data,
    )
    
    from .validators import (
        AllocationValidator,
        FairnessValidator,
        DataValidator,
        validate_allocation_matrix,
        validate_fairness_constraints,
        validate_zk_proof,
    )
    
except ImportError as e:
    warnings.warn(f"Data handling modules not available: {e}")
    
    def load_dataset(name: str):
        raise ImportError("Data modules not available")

# Domain-specific applications
try:
    from .domains.housing import (
        HousingAllocationSystem,
        HousingDataset,
        BerlinHousingMarket,
        AccessibilityConstraints,
        GeographicFairness,
    )
    
    from .domains.healthcare import (
        HealthcareAllocationSystem,
        HospitalBedAllocator,
        MedicalPrioritySystem,
        PatientQueueManager,
        TriageAllocator,
    )
    
    from .domains.general import (
        GeneralAllocationSystem,
        ResourcePool,
        AgentPopulation,
        ConstraintSystem,
    )
    
except ImportError as e:
    warnings.warn(f"Domain-specific modules not available: {e}")

# Simulation and experimentation
try:
    from .simulation import (
        AllocationSimulator,
        ConvergenceAnalyzer,
        PerformanceBenchmark,
        ExperimentRunner,
        run_simulation,
        analyze_convergence,
        benchmark_algorithms,
    )
    
    from .experiments import (
        ReproducibilityFramework,
        FairnessExperiment,
        ScalabilityExperiment,
        TemporalConsistencyExperiment,
    )
    
except ImportError as e:
    warnings.warn(f"Simulation modules not available: {e}")

# Utilities and helpers
try:
    from .utils import (
        Logger,
        Timer,
        MemoryProfiler,
        ConfigManager,
        ResultsManager,
        create_logger,
        load_config,
        save_results,
    )
    
    from .exceptions import (
        AletheiaError,
        AllocationError,
        ConvergenceError,
        FairnessViolationError,
        ZKProofError,
        DataValidationError,
        ConfigurationError,
    )
    
except ImportError as e:
    warnings.warn(f"Utility modules not available: {e}")

# Lean 4 formal verification interface
try:
    from .verification import (
        LeanInterface,
        TheoremProver,
        ProofChecker,
        verify_lean_proofs,
        check_theorem,
        export_to_lean,
    )
    
except ImportError as e:
    warnings.warn(f"Formal verification interface not available: {e}")

# Configuration and logging setup
try:
    from .config import (
        DEFAULT_CONFIG,
        AlgorithmConfig,
        ZKConfig,
        ExperimentConfig,
        load_configuration,
        create_default_config,
    )
    
    # Initialize default configuration
    _default_config = create_default_config()
    
except ImportError as e:
    warnings.warn(f"Configuration system not available: {e}")
    _default_config = {}

# Package-level constants
SUPPORTED_ALGORITHMS = [
    "alpha_fairness",
    "proportional_fairness", 
    "max_min_fairness",
    "entropy_duality",
    "weighted_fairness",
]

SUPPORTED_ZK_BACKENDS = [
    "groth16",
    "halo2", 
    "plonk",
]

SUPPORTED_DOMAINS = [
    "housing",
    "healthcare",
    "general",
]

# Main API functions
def get_version() -> str:
    """Get the current version of Project Aletheia."""
    return __version__

def get_info() -> Dict[str, Any]:
    """Get comprehensive package information."""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "url": __url__,
        "license": __license__,
        "status": __status__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "supported_algorithms": SUPPORTED_ALGORITHMS,
        "supported_zk_backends": SUPPORTED_ZK_BACKENDS,
        "supported_domains": SUPPORTED_DOMAINS,
    }

def create_allocator(algorithm: str = "alpha_fairness", **kwargs) -> Any:
    """
    Create an allocation algorithm instance.
    
    Args:
        algorithm: Algorithm name from SUPPORTED_ALGORITHMS
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Allocator instance
        
    Example:
        >>> allocator = ale.create_allocator("alpha_fairness", alpha=1.0)
        >>> result = allocator.allocate(data)
    """
    algorithm_map = {
        "alpha_fairness": AlphaFairnessAllocator,
        "proportional_fairness": ProportionalFairnessAllocator,
        "max_min_fairness": MaxMinFairnessAllocator,
        "entropy_duality": EntropyDualityAllocator,
        "weighted_fairness": WeightedFairnessAllocator,
    }
    
    if algorithm not in algorithm_map:
        raise ValueError(f"Unsupported algorithm: {algorithm}. "
                        f"Supported algorithms: {SUPPORTED_ALGORITHMS}")
    
    return algorithm_map[algorithm](**kwargs)

def create_zk_system(backend: str = "groth16", **kwargs) -> Any:
    """
    Create a zero-knowledge proof system.
    
    Args:
        backend: ZK backend from SUPPORTED_ZK_BACKENDS
        **kwargs: Backend-specific parameters
        
    Returns:
        ZK proof system instance
    """
    backend_map = {
        "groth16": Groth16Backend,
        "halo2": Halo2Backend,
        "plonk": PlonkBackend,
    }
    
    if backend not in backend_map:
        raise ValueError(f"Unsupported ZK backend: {backend}. "
                        f"Supported backends: {SUPPORTED_ZK_BACKENDS}")
    
    return ZKProofSystem(backend=backend_map[backend](**kwargs))

def quick_allocation(
    data: Any,
    algorithm: str = "alpha_fairness",
    generate_proof: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick allocation with sensible defaults.
    
    Args:
        data: Allocation problem data
        algorithm: Algorithm to use
        generate_proof: Whether to generate ZK proof
        **kwargs: Additional parameters
        
    Returns:
        Dict containing allocation results and optional proof
        
    Example:
        >>> data = ale.load_dataset("berlin_housing") 
        >>> result = ale.quick_allocation(data, generate_proof=True)
        >>> print(f"Gini coefficient: {result['metrics']['gini']}")
    """
    # Create allocator
    allocator = create_allocator(algorithm, **kwargs)
    
    # Run allocation
    allocation_result = allocator.allocate(data)
    
    # Prepare result dictionary
    result = {
        "allocation": allocation_result.allocations,
        "metrics": {
            "gini_coefficient": compute_gini_coefficient(allocation_result),
            "theil_index": compute_theil_index(allocation_result),
            "envy_ratio": compute_envy_ratio(allocation_result),
            "utilitarian_welfare": compute_utilitarian_welfare(allocation_result),
        },
        "convergence": allocation_result.convergence_info,
        "fairness_satisfied": allocation_result.fairness_constraints_satisfied,
    }
    
    # Generate ZK proof if requested
    if generate_proof:
        try:
            zk_system = create_zk_system()
            proof = zk_system.generate_proof(allocation_result)
            result["zk_proof"] = proof
            result["proof_verified"] = zk_system.verify_proof(proof)
        except ImportError:
            warnings.warn("ZK proof generation requested but not available")
            result["zk_proof"] = None
            result["proof_verified"] = False
    
    return result

def run_experiment(
    experiment_type: str,
    datasets: list,
    algorithms: Optional[list] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a predefined experiment type.
    
    Args:
        experiment_type: Type of experiment to run
        datasets: List of datasets to test on
        algorithms: List of algorithms to compare
        **kwargs: Experiment-specific parameters
        
    Returns:
        Comprehensive experiment results
    """
    if algorithms is None:
        algorithms = ["alpha_fairness", "proportional_fairness"]
    
    experiment_map = {
        "fairness": FairnessExperiment,
        "scalability": ScalabilityExperiment, 
        "temporal": TemporalConsistencyExperiment,
    }
    
    if experiment_type not in experiment_map:
        raise ValueError(f"Unsupported experiment type: {experiment_type}")
    
    experiment = experiment_map[experiment_type](
        datasets=datasets,
        algorithms=algorithms,
        **kwargs
    )
    
    return experiment.run()

# Compatibility and migration helpers
def migrate_from_v1(old_config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate configuration from v1.x to current version."""
    warnings.warn("Migration from v1.x is deprecated", DeprecationWarning)
    # Migration logic would go here
    return old_config

# Package initialization
def _initialize_package():
    """Initialize package-level settings and configurations."""
    try:
        # Set up logging
        logger = create_logger("aletheia", level="INFO")
        logger.info(f"Initialized Project Aletheia v{__version__}")
        
        # Validate environment
        if "ALETHEIA_CONFIG" in os.environ:
            config_path = Path(os.environ["ALETHEIA_CONFIG"])
            if config_path.exists():
                global _default_config
                _default_config = load_configuration(config_path)
                logger.info(f"Loaded configuration from {config_path}")
        
        # Check for optional dependencies
        missing_deps = []
        try:
            import cvxpy
        except ImportError:
            missing_deps.append("cvxpy (optimization)")
            
        try:
            import galois
        except ImportError:
            missing_deps.append("galois (finite field arithmetic)")
            
        if missing_deps:
            logger.warning(f"Optional dependencies not available: {', '.join(missing_deps)}")
            
    except Exception as e:
        warnings.warn(f"Package initialization warning: {e}")

# Initialize package
import os
try:
    _initialize_package()
except Exception as e:
    warnings.warn(f"Failed to initialize package: {e}")

# Public API - what gets imported with "from aletheia import *"
__all__ = [
    # Version and metadata
    "__version__",
    "__author__", 
    "__description__",
    
    # Core allocator classes
    "AlphaFairnessAllocator",
    "ProportionalFairnessAllocator", 
    "MaxMinFairnessAllocator",
    "EntropyDualityAllocator",
    
    # ZK proof functions
    "generate_zk_proof",
    "verify_zk_proof",
    "ZKProofSystem",
    
    # Data and validation
    "load_dataset",
    "validate_dataset", 
    "Dataset",
    
    # Metrics
    "compute_gini_coefficient",
    "compute_theil_index",
    "compute_envy_ratio",
    "FairnessAnalyzer",
    
    # Simulation
    "run_simulation",
    "AllocationSimulator",
    
    # Domain applications
    "HousingAllocationSystem",
    "HealthcareAllocationSystem",
    
    # Main API functions
    "get_version",
    "get_info",
    "create_allocator",
    "create_zk_system", 
    "quick_allocation",
    "run_experiment",
    
    # Constants
    "SUPPORTED_ALGORITHMS",
    "SUPPORTED_ZK_BACKENDS",
    "SUPPORTED_DOMAINS",
    
    # Exceptions
    "AletheiaError",
    "AllocationError",
    "ZKProofError",
]

# Cleanup - remove internal variables from public namespace
del sys, warnings, Path, Dict, Any, Optional
