# Contributing to Quantum-Neural-Search

Thanks for your interest in contributing. This document covers the setup, conventions, and workflow for the project.

## Setup

```bash
git clone https://github.com/IsolatedSingularity/Quantum-Neural-Search.git
cd Quantum-Neural-Search
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Code Conventions

- **camelCase everywhere.** All variables, functions, identifiers. This overrides PEP 8.
- Type annotations on all public functions.
- Imports: stdlib first, then third-party (one per line), then local.
- Line length: 100 characters max.

## Linting and Formatting

```bash
ruff check .          # Lint
ruff format .         # Auto-format
mypy quantumNeuralSearch/ --ignore-missing-imports   # Type check
```

All three must pass before committing. A `.pre-commit-config.yaml` is included; install hooks with:

```bash
pip install pre-commit
pre-commit install
```

## Running Tests

```bash
pytest                # Runs all tests with coverage
pytest --no-cov -q    # Quick run without coverage
```

Tests live in `tests/`. Naming convention: `test<Module>.py` with `test_<description>` functions.

## Project Structure

- `quantumNeuralSearch/`: Core library (Grover search, VQC, encodings, brain atlas, connectivity, dynamics)
- `quantumNeuralSearch/visualization/`: Plot generation modules
- `tests/`: pytest test suite
- `scripts/`: Utility scripts (headless plot generation)
- `Plots/`: Generated output figures (committed for README display)

## Workflow

1. Fork and create a feature branch.
2. Make changes following the conventions above.
3. Run `ruff check .`, `ruff format --check .`, `mypy`, and `pytest`.
4. Open a PR against `main`.

CI runs lint, type check, and tests on Python 3.11 and 3.12 automatically.

## Forbidden Language (Public-Facing Content)

When editing README, docstrings, or descriptions:

- Zero em dashes. Use commas, colons, semicolons, or parentheses.
- No buzzword stacking (max one domain-specific term per sentence).
- Banned phrases: "architected", "leveraged", "enterprise", "translated X into Y", "delve", "I would love to", "Let's dive in", "revolutionize".
