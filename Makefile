# Project Aletheia - α-fair, Cryptographically-Auditable Allocation Framework
# Makefile for building, testing, and reproducing experiments
#
# Usage:
#   make env      - Set up development environment
#   make test     - Run all tests
#   make reproduce - Run full replication pipeline
#   make clean    - Clean build artifacts
#   make help     - Show detailed help

.PHONY: help env build test clean reproduce docs format lint typecheck
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
PIP := pip
VENV_DIR := venv
RUST_TARGET := target
LEAN_BUILD := proofs/build
ZK_PROOFS := zk/sample_proofs
EXPERIMENTS := experiments/results
DOCS_BUILD := docs/_build

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m # No Color

# Version information
VERSION := $(shell grep "version" pyproject.toml | head -1 | cut -d '"' -f 2)
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")

# Help target
help: ## Show this help message
	@echo "$(CYAN)Project Aletheia - α-fair, Cryptographically-Auditable Allocation Framework$(NC)"
	@echo "$(BLUE)Version: $(VERSION) | Commit: $(GIT_COMMIT) | Built: $(BUILD_DATE)$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  $(GREEN)make env$(NC)        # Set up development environment"
	@echo "  $(GREEN)make test$(NC)       # Run test suite"
	@echo "  $(GREEN)make reproduce$(NC)  # Run full replication"
	@echo ""
	@echo "$(YELLOW)Documentation:$(NC) See docs/replication.md for detailed instructions"

# Environment Setup
env: ## Set up complete development environment
	@echo "$(CYAN)Setting up Project Aletheia development environment...$(NC)"
	@$(MAKE) setup-python
	@$(MAKE) setup-rust
	@$(MAKE) setup-lean
	@$(MAKE) install-deps
	@$(MAKE) verify-setup
	@echo "$(GREEN)✓ Environment setup complete!$(NC)"

setup-python: ## Set up Python virtual environment
	@echo "$(BLUE)Setting up Python environment...$(NC)"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@. $(VENV_DIR)/bin/activate && \
		$(PIP) install --upgrade pip setuptools wheel

setup-rust: ## Set up Rust toolchain
	@echo "$(BLUE)Setting up Rust toolchain...$(NC)"
	@if ! command -v rustc > /dev/null 2>&1; then \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
	fi
	@. $$HOME/.cargo/env && \
		rustup install stable && \
		rustup default stable && \
		rustup component add clippy rustfmt

setup-lean: ## Set up Lean 4 theorem prover
	@echo "$(BLUE)Setting up Lean 4 theorem prover...$(NC)"
	@if ! command -v lean > /dev/null 2>&1; then \
		curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y; \
	fi
	@. $$HOME/.elan/env && \
		elan install leanprover/lean4:stable && \
		elan default leanprover/lean4:stable

install-deps: ## Install all project dependencies
	@echo "$(BLUE)Installing project dependencies...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PIP) install -e .
	@. $$HOME/.cargo/env && \
		cd zk && cargo fetch
	@. $$HOME/.elan/env && \
		cd proofs && lake exe cache get

verify-setup: ## Verify development environment is correctly configured
	@echo "$(BLUE)Verifying environment setup...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) -c "import aletheia; print('✓ Python environment OK')"
	@. $$HOME/.cargo/env && \
		cargo --version > /dev/null && echo "✓ Rust toolchain OK"
	@. $$HOME/.elan/env && \
		lean --version > /dev/null && echo "✓ Lean 4 OK"
	@echo "$(GREEN)✓ All components verified$(NC)"

# Build targets
build: build-python build-rust build-lean ## Build all components

build-python: ## Build Python components
	@echo "$(BLUE)Building Python components...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) -m build

build-rust: ## Build Rust components
	@echo "$(BLUE)Building Rust components...$(NC)"
	@. $$HOME/.cargo/env && \
		cd zk && cargo build --release

build-lean: ## Build Lean proofs
	@echo "$(BLUE)Building Lean proofs...$(NC)"
	@. $$HOME/.elan/env && \
		cd proofs && lake build

# Testing targets
test: test-unit test-integration test-proofs test-zk ## Run all tests

test-unit: ## Run unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) -m pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) -m pytest tests/integration/ -v --tb=short

test-proofs: ## Verify Lean proofs
	@echo "$(BLUE)Verifying Lean proofs...$(NC)"
	@. $$HOME/.elan/env && \
		cd proofs && lean --make src/ergodicity.lean

test-zk: ## Test zero-knowledge proof components
	@echo "$(BLUE)Testing ZK proof components...$(NC)"
	@. $$HOME/.cargo/env && \
		cd zk && cargo test --release

test-quick: ## Run quick test suite (unit tests only)
	@echo "$(BLUE)Running quick test suite...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) -m pytest tests/unit/ -x --tb=line

test-algorithms: ## Test allocation algorithms specifically
	@echo "$(BLUE)Testing allocation algorithms...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) -m pytest tests/unit/algorithms/ -v

test-fairness: ## Test fairness metrics and properties
	@echo "$(BLUE)Testing fairness components...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) -m pytest tests/unit/fairness/ -v

# Code quality targets
format: format-python format-rust format-lean ## Format all code

format-python: ## Format Python code with black
	@echo "$(BLUE)Formatting Python code...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		black python/ tests/ --line-length 88

format-rust: ## Format Rust code with rustfmt
	@echo "$(BLUE)Formatting Rust code...$(NC)"
	@. $$HOME/.cargo/env && \
		cd zk && cargo fmt

format-lean: ## Format Lean code
	@echo "$(BLUE)Formatting Lean code...$(NC)"
	@. $$HOME/.elan/env && \
		cd proofs && find src -name "*.lean" -exec lean --stdin < {} \;

lint: lint-python lint-rust ## Run linting on all code

lint-python: ## Lint Python code
	@echo "$(BLUE)Linting Python code...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		flake8 python/ tests/ --max-line-length=88 --extend-ignore=E203,W503

lint-rust: ## Lint Rust code with clippy
	@echo "$(BLUE)Linting Rust code...$(NC)"
	@. $$HOME/.cargo/env && \
		cd zk && cargo clippy --all-targets --all-features -- -D warnings

typecheck: ## Run type checking on Python code
	@echo "$(BLUE)Running type checks...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		mypy python/ --ignore-missing-imports

format-check: ## Check if code formatting is correct (CI)
	@echo "$(BLUE)Checking code formatting...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		black python/ tests/ --check --line-length 88

# Data and experiment targets
data-generate: ## Generate sample datasets
	@echo "$(BLUE)Generating sample datasets...$(NC)"
	@mkdir -p data/sample data/scenarios
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) python/alethia/data_generator.py --city berlin --output data/sample/berlin_sample.json

data-validate: ## Validate existing datasets
	@echo "$(BLUE)Validating datasets...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) python/alethia/validators.py --validate-data data/sample/berlin_sample.json

# Simulation and experiment targets
simulate: ## Run basic allocation simulation
	@echo "$(BLUE)Running allocation simulation...$(NC)"
	@mkdir -p $(EXPERIMENTS)
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) python/alethia/simulator.py --config configs/berlin_config.json --output $(EXPERIMENTS)/sample_run.json

simulate-batch: ## Run batch simulations for different scenarios
	@echo "$(BLUE)Running batch simulations...$(NC)"
	@mkdir -p $(EXPERIMENTS)
	@. $(VENV_DIR)/bin/activate && \
		for scenario in fairness convergence multi_district; do \
			$(PYTHON) python/alethia/simulator.py --config configs/$${scenario}_config.json --output $(EXPERIMENTS)/$${scenario}_results.json; \
		done

# Zero-knowledge proof targets
zk-generate: ## Generate sample ZK proofs
	@echo "$(BLUE)Generating ZK proofs...$(NC)"
	@mkdir -p $(ZK_PROOFS)
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) python/alethia/zk_interface.py --generate-proof --input data/sample/berlin_sample.json --output $(ZK_PROOFS)/sample.proof

zk-verify: ## Verify ZK proofs
	@echo "$(BLUE)Verifying ZK proofs...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) python/alethia/zk_interface.py --verify-proof --input $(ZK_PROOFS)/sample.proof

zk-benchmark: ## Benchmark ZK proof performance
	@echo "$(BLUE)Benchmarking ZK proof performance...$(NC)"
	@. $$HOME/.cargo/env && \
		cd zk && ./bench.sh --performance-test

# Documentation targets
docs: docs-build docs-api ## Build all documentation

docs-build: ## Build documentation with Sphinx
	@echo "$(BLUE)Building documentation...$(NC)"
	@mkdir -p $(DOCS_BUILD)
	@. $(VENV_DIR)/bin/activate && \
		cd docs && make html

docs-api: ## Generate API documentation
	@echo "$(BLUE)Generating API documentation...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		sphinx-apidoc -o docs/api python/alethia --force

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		cd $(DOCS_BUILD)/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	@echo "$(BLUE)Cleaning documentation build...$(NC)"
	@rm -rf $(DOCS_BUILD)

# Performance and benchmarking
benchmark: benchmark-algorithms benchmark-zk ## Run all benchmarks

benchmark-algorithms: ## Benchmark allocation algorithms
	@echo "$(BLUE)Benchmarking allocation algorithms...$(NC)"
	@mkdir -p $(EXPERIMENTS)/benchmarks
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) python/alethia/benchmarks.py --suite allocation_performance --output $(EXPERIMENTS)/benchmarks/algorithms.json

benchmark-zk: ## Benchmark zero-knowledge proofs
	@echo "$(BLUE)Benchmarking ZK proofs...$(NC)"
	@mkdir -p $(EXPERIMENTS)/benchmarks
	@. $$HOME/.cargo/env && \
		cd zk && ./bench.sh --benchmark --output ../$(EXPERIMENTS)/benchmarks/zk_performance.json

# Main replication pipeline
reproduce: ## Run complete replication pipeline
	@echo "$(CYAN)Starting Project Aletheia replication pipeline...$(NC)"
	@echo "$(YELLOW)This will take approximately 15-30 minutes$(NC)"
	@$(MAKE) clean-results
	@$(MAKE) data-generate
	@$(MAKE) test-proofs
	@$(MAKE) simulate
	@$(MAKE) zk-generate
	@$(MAKE) zk-verify
	@$(MAKE) validate-results
	@$(MAKE) generate-report
	@echo "$(GREEN)✓ Replication pipeline completed successfully!$(NC)"
	@echo "$(YELLOW)Results available in: $(EXPERIMENTS)$(NC)"

validate-results: ## Validate experimental results
	@echo "$(BLUE)Validating experimental results...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) python/alethia/validators.py --validate-results $(EXPERIMENTS)/sample_run.json

generate-report: ## Generate replication report
	@echo "$(BLUE)Generating replication report...$(NC)"
	@mkdir -p $(EXPERIMENTS)/reports
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) python/alethia/report_generator.py --input $(EXPERIMENTS) --output $(EXPERIMENTS)/reports/replication_report.pdf

# Development utilities
dev-setup: env ## Set up complete development environment with additional tools
	@echo "$(BLUE)Installing additional development tools...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PIP) install pytest-cov pytest-xdist black flake8 mypy sphinx sphinx-rtd-theme
	@echo "$(GREEN)✓ Development setup complete$(NC)"

watch-tests: ## Watch for changes and run tests automatically
	@echo "$(BLUE)Watching for changes and running tests...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) -m pytest_watch -- tests/unit/

profile: ## Profile algorithm performance
	@echo "$(BLUE)Profiling algorithm performance...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) -m cProfile -o $(EXPERIMENTS)/profile.stats python/alethia/simulator.py --config configs/berlin_config.json

memory-profile: ## Profile memory usage
	@echo "$(BLUE)Profiling memory usage...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		mprof run python/alethia/simulator.py --config configs/berlin_config.json && \
		mprof plot -o $(EXPERIMENTS)/memory_profile.png

# Security and analysis
security-scan: ## Run security analysis
	@echo "$(BLUE)Running security analysis...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		bandit -r python/ -f json -o $(EXPERIMENTS)/security_report.json || true
	@. $$HOME/.cargo/env && \
		cd zk && cargo audit || true

dependency-check: ## Check for dependency vulnerabilities
	@echo "$(BLUE)Checking dependencies for vulnerabilities...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		safety check --json --output $(EXPERIMENTS)/dependency_report.json || true

# CI/CD targets
ci-setup: ## Set up environment for CI/CD
	@$(MAKE) env
	@$(MAKE) format-check
	@$(MAKE) lint
	@$(MAKE) typecheck

ci-test: ## Run tests in CI environment
	@$(MAKE) test-unit
	@$(MAKE) test-integration
	@$(MAKE) test-proofs

ci-build: ## Build artifacts for CI
	@$(MAKE) build
	@$(MAKE) docs-build

# Deployment targets
package: ## Package for distribution
	@echo "$(BLUE)Packaging for distribution...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) -m build
	@. $$HOME/.cargo/env && \
		cd zk && cargo package

release-check: ## Check if ready for release
	@echo "$(BLUE)Checking release readiness...$(NC)"
	@$(MAKE) test
	@$(MAKE) format-check
	@$(MAKE) lint
	@$(MAKE) docs-build
	@$(MAKE) security-scan
	@echo "$(GREEN)✓ Release checks passed$(NC)"

# Cleaning targets
clean: clean-build clean-cache clean-results ## Clean all generated files

clean-build: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@rm -rf $(RUST_TARGET)
	@rm -rf $(LEAN_BUILD)
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true

clean-cache: ## Clean cache files
	@echo "$(BLUE)Cleaning cache files...$(NC)"
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@find . -name ".DS_Store" -delete 2>/dev/null || true

clean-results: ## Clean experimental results
	@echo "$(BLUE)Cleaning experimental results...$(NC)"
	@rm -rf $(EXPERIMENTS)
	@rm -rf $(ZK_PROOFS)
	@mkdir -p $(EXPERIMENTS) $(ZK_PROOFS)

clean-docs: docs-clean ## Alias for docs-clean

deep-clean: clean ## Complete clean including virtual environment
	@echo "$(BLUE)Performing deep clean...$(NC)"
	@rm -rf $(VENV_DIR)
	@echo "$(YELLOW)Note: Run 'make env' to recreate environment$(NC)"

# Docker targets
docker-build: ## Build Docker development image
	@echo "$(BLUE)Building Docker development image...$(NC)"
	@docker build -t aletheia-dev -f Dockerfile.dev .

docker-run: ## Run development container
	@echo "$(BLUE)Running development container...$(NC)"
	@docker run -it --rm -v $(PWD):/workspace aletheia-dev

docker-test: ## Run tests in Docker container
	@echo "$(BLUE)Running tests in Docker container...$(NC)"
	@docker run --rm -v $(PWD):/workspace aletheia-dev make test

# Information targets
info: ## Show project information
	@echo "$(CYAN)Project Aletheia Information$(NC)"
	@echo "$(BLUE)Version:$(NC) $(VERSION)"
	@echo "$(BLUE)Git Commit:$(NC) $(GIT_COMMIT)"
	@echo "$(BLUE)Build Date:$(NC) $(BUILD_DATE)"
	@echo "$(BLUE)Python Version:$(NC) $(shell $(PYTHON) --version 2>&1)"
	@echo "$(BLUE)Repository:$(NC) https://github.com/samansiddiqui55/Aletheia"

status: ## Show environment status
	@echo "$(CYAN)Environment Status$(NC)"
	@echo -n "$(BLUE)Python Environment:$(NC) "
	@if [ -d "$(VENV_DIR)" ]; then echo "$(GREEN)✓ Active$(NC)"; else echo "$(RED)✗ Not set up$(NC)"; fi
	@echo -n "$(BLUE)Rust Toolchain:$(NC) "
	@if command -v rustc > /dev/null 2>&1; then echo "$(GREEN)✓ Available$(NC)"; else echo "$(RED)✗ Not installed$(NC)"; fi
	@echo -n "$(BLUE)Lean 4:$(NC) "
	@if command -v lean > /dev/null 2>&1; then echo "$(GREEN)✓ Available$(NC)"; else echo "$(RED)✗ Not installed$(NC)"; fi
	@echo -n "$(BLUE)Build Artifacts:$(NC) "
	@if [ -d "$(RUST_TARGET)" ] && [ -d "$(LEAN_BUILD)" ]; then echo "$(GREEN)✓ Present$(NC)"; else echo "$(YELLOW)⚠ Missing$(NC)"; fi

list-deps: ## List project dependencies
	@echo "$(CYAN)Project Dependencies$(NC)"
	@echo "$(BLUE)Python packages:$(NC)"
	@. $(VENV_DIR)/bin/activate && $(PIP) list
	@echo "$(BLUE)Rust crates:$(NC)"
	@. $$HOME/.cargo/env && cd zk && cargo tree --depth 1

# Experimental and research targets
experiment-fairness: ## Run fairness analysis experiments
	@echo "$(BLUE)Running fairness analysis experiments...$(NC)"
	@mkdir -p $(EXPERIMENTS)/fairness
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) experiments/scripts/fairness_analysis.py --output $(EXPERIMENTS)/fairness/

experiment-convergence: ## Run convergence analysis experiments  
	@echo "$(BLUE)Running convergence analysis experiments...$(NC)"
	@mkdir -p $(EXPERIMENTS)/convergence
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) experiments/scripts/convergence_analysis.py --output $(EXPERIMENTS)/convergence/

experiment-scalability: ## Run scalability analysis experiments
	@echo "$(BLUE)Running scalability analysis experiments...$(NC)"
	@mkdir -p $(EXPERIMENTS)/scalability
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) experiments/scripts/scalability_analysis.py --output $(EXPERIMENTS)/scalability/

research-report: ## Generate comprehensive research report
	@echo "$(BLUE)Generating research report...$(NC)"
	@$(MAKE) experiment-fairness
	@$(MAKE) experiment-convergence  
	@$(MAKE) experiment-scalability
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) experiments/scripts/generate_research_report.py --input $(EXPERIMENTS) --output $(EXPERIMENTS)/research_report.pdf

# Debugging and troubleshooting
debug-env: ## Debug environment issues
	@echo "$(CYAN)Environment Debug Information$(NC)"
	@echo "$(BLUE)Operating System:$(NC) $(shell uname -a)"
	@echo "$(BLUE)Make Version:$(NC) $(shell make --version | head -1)"
	@echo "$(BLUE)Python Path:$(NC) $(shell which $(PYTHON))"
	@echo "$(BLUE)Python Version:$(NC) $(shell $(PYTHON) --version 2>&1)"
	@echo "$(BLUE)Virtual Environment:$(NC) $(VENV_DIR)"
	@echo "$(BLUE)Current Directory:$(NC) $(PWD)"
	@echo "$(BLUE)Git Status:$(NC)"
	@git status --porcelain || echo "Not a git repository"

debug-deps: ## Debug dependency issues
	@echo "$(CYAN)Dependency Debug Information$(NC)"
	@echo "$(BLUE)Python packages in virtual environment:$(NC)"
	@. $(VENV_DIR)/bin/activate && $(PIP) check || true
	@echo "$(BLUE)Rust cargo check:$(NC)"
	@. $$HOME/.cargo/env && cd zk && cargo check || true
	@echo "$(BLUE)Lean dependencies:$(NC)"
	@. $$HOME/.elan/env && cd proofs && lake exe cache get --update || true

debug-tests: ## Debug test failures
	@echo "$(CYAN)Test Debug Information$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTHON) -m pytest --collect-only tests/ | grep "collected" || echo "No tests found"
	@echo "$(BLUE)Test configuration:$(NC)"
	@cat pytest.ini 2>/dev/null || echo "No pytest.ini found"

# Special targets for specific use cases
academic-setup: ## Set up environment for academic research
	@$(MAKE) env
	@$(MAKE) docs-build
	@echo "$(GREEN)✓ Academic research environment ready$(NC)"
	@echo "$(YELLOW)Recommended next steps:$(NC)"
	@echo "  1. Review docs/theory/ for mathematical foundations"
	@echo "  2. Run 'make experiment-fairness' for fairness analysis"
	@echo "  3. Explore proofs/ directory for formal verification"

industry-setup: ## Set up environment for industry deployment
	@$(MAKE) env
	@$(MAKE) security-scan
	@$(MAKE) benchmark
	@echo "$(GREEN)✓ Industry deployment environment ready$(NC)"
	@echo "$(YELLOW)Recommended next steps:$(NC)"
	@echo "  1. Review security_report.json for security analysis"
	@echo "  2. Run 'make benchmark' for performance evaluation"
	@echo "  3. Consider 'make docker-build' for containerized deployment"

teaching-setup: ## Set up environment for educational use
	@$(MAKE) env
	@$(MAKE) docs-build
	@$(MAKE) data-generate
	@echo "$(GREEN)✓ Educational environment ready$(NC)"
	@echo "$(YELLOW)Recommended resources:$(NC)"
	@echo "  1. docs/tutorials/ - Step-by-step tutorials"
	@echo "  2. examples/ - Working code examples"  
	@echo "  3. data/sample/ - Sample datasets for experimentation"

# Version and release management
version: ## Show current version
	@echo "$(VERSION)"

bump-version: ## Bump version (use VERSION=x.y.z)
	@if [ -z "$(VERSION)" ]; then \
		echo "$(RED)Error: Please specify VERSION=x.y.z$(NC)"; \
		exit 1; \
	fi
	@sed -i 's/version = "[^"]*"/version = "$(VERSION)"/' pyproject.toml
	@sed -i 's/version: "[^"]*"/version: "$(VERSION)"/' CITATION.cff
	@echo "$(GREEN)✓ Version bumped to $(VERSION)$(NC)"

tag-release: ## Tag current version for release
	@git tag -a "v$(VERSION)" -m "Release version $(VERSION)"
	@echo "$(GREEN)✓ Tagged release v$(VERSION)$(NC)"

# Integration with external tools
jupyter-setup: ## Set up Jupyter environment for interactive development
	@echo "$(BLUE)Setting up Jupyter environment...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PIP) install jupyter ipykernel matplotlib seaborn && \
		$(PYTHON) -m ipykernel install --user --name aletheia --display-name "Aletheia"
	@echo "$(GREEN)✓ Jupyter environment ready$(NC)"
	@echo "$(YELLOW)Start with: jupyter notebook$(NC)"

vscode-setup: ## Set up VS Code configuration
	@echo "$(BLUE)Setting up VS Code configuration...$(NC)"
	@mkdir -p .vscode
	@echo '{"python.defaultInterpreterPath": "./venv/bin/python", "rust-analyzer.server.path": "rust-analyzer"}' > .vscode/settings.json
	@echo "$(GREEN)✓ VS Code configuration created$(NC)"

# Final catch-all and error handling
.PHONY: $(MAKECMDGOALS)

# Error handling for missing targets
%:
	@echo "$(RED)Error: Target '$@' not found$(NC)"
	@echo "$(YELLOW)Available targets:$(NC)"
	@$(MAKE) help
	@exit 1
