# redMaPPer Project Overview

The **red**-sequence **ma**tched-filter **P**robabilistic **Per**colation (**redMaPPer**) cluster finder is a Python implementation of the algorithm used for detecting galaxy clusters in large photometric surveys. It is based on the red-sequence of cluster galaxies and uses a matched-filter approach to identify clusters and estimate their richness and redshift.

## Technology Stack

- **Language:** Python 3.7+ (with C extensions for performance-critical tasks).
- **Package Manager:** [uv](https://github.com/astral-sh/uv). Use `uv run python` for execution and `uv add`/`uv remove` for dependencies.
- **Core Libraries:** `numpy`, `scipy`, `astropy`, `matplotlib`, `pyyaml`.
- **Data Formats:** FITS (via `fitsio` and `esutil`), HEALSparse (for masks and depth maps).
- **Spatial Indexing:** HEALPix (via `hpgeom` and `healsparse`).
- **Configuration:** YAML-based configuration files.
- **Task Management:** Support for local multiprocessing and batch systems (e.g., LSF).

## Architectural Direction & Refactoring

The project is currently undergoing a **radical architectural refactoring** to transition from a legacy Object-Oriented Programming (OOP) approach to a **Functional Programming (FP)** style.

- **Status:** The transition is approximately halfway complete.
- **Goal:** Minimize coupling and improve testability through pure functions and immutable data structures where possible.
- **Mandate:** Implementation changes should be **complete and radical**. There is **no requirement for backward compatibility** with the old OOP structures during this refactoring.
- **Development Priority:** Prioritize explicit composition and delegation over inheritance.

## Pipeline Structure

The whole pipeline is primarily composed of two main stages: **Calibration** and **Finder / redmapper_run**.

1.  **Calibration:** The entry point for the calibration suite is `bin/redmapper_calibrate.py`.
2.  **Finder / redmapper_run:** The primary logic for cluster finding.

## Key Components

- **`redmapper.configuration.Configuration`**: Centralized configuration management.
- **`redmapper.catalog.GalaxyCatalog`**: Interfaces for reading and handling galaxy and cluster catalogs.
- **`redmapper.pipeline`**: Task-based system for running different stages of the cluster finder.
- **`redmapper.calibration`**: Modules for calibrating the red-sequence, centering, and richness.
- **`redmapper.redmagic`**: Implementation of the redMaGiC selector.
- **C Extensions**:
  - `redmapper/chisq_dist/`: Fast chi-square distribution calculations.
  - `redmapper/solver_nfw/`: Fast NFW profile weight calculations.

## Building and Running

### Execution

Always use `uv` for consistent environment management:

```bash
uv run python <script.py>
```

### Dependency Management

```bash
uv add <package>
uv remove <package>
```

### Running Tests

The project uses **pytest** for its test suite (nosetests is deprecated for this project). Note that tests must be run from within the `tests/` directory:

```bash
cd tests
uv run pytest
```

## Development Conventions

- **Configuration:** Almost all operations require a YAML configuration file. Use `redmapper.Configuration` to load and access parameters.
- **Data Handling:** Prefer `fitsio` for efficient FITS I/O.
- **Logging:** Use the custom logger in `redmapper.logger`.
- **Spatial Logic:** HEALPix is used for all spatial calculations. Masks and depth maps are handled via `healsparse`.
- **Concurrency:** The pipeline supports both local multiprocessing (`config.calib_nproc`) and distributed runs via HEALPix pixelization.

## Directory Structure Highlights

- `redmapper/`: Core package containing algorithm implementations.
- `bin/`: Executable scripts for end-to-end workflows.
- `how-to/`: Documentation and examples for running the code.
- `tests/`: Comprehensive test suite and sample data for testing.
- `docs/`: Sphinx-based documentation.
