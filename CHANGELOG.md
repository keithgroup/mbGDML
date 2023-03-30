# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Iterative solver for GDML training on larger systems (changes up to v1.0.0).
- Additional documentation on obtaining many-body data.
- Some many-body expansion logging (for debugging).
- Logger documentation.
- Function to change mbGDML log levels.
- Ability of ASE calculator to update periodic cell on ``mbePredict`` object.
- Preliminary virial stress algorithms and ``stress`` module.
- Switching function module for alchemical scaling.
- Documentation example of an ASE optimization under periodic boundary conditions.
- Structure generation module using packmol.
- Provide explicit example of model comp ids in documentation.

### Changed

- GDML solvers are put into their own modules.
- Updated prediction set documentation.
- Unified use of ray by always having ``use_ray`` option and defaulting ``n_workers`` to ``1``.

### Fixed

- `utils.center_structures` sometimes repeated the array incorrectly.
- Explicitly use 3DMol v1.8.0 for documentation.

## [0.0.4] - 2022-12-12

### Added

- ``mbgdml.descriptors.Criteria`` class for setting up structure descriptors and cutoffs for models.
- Model analysis with Matérn covariance function.
- ``mbe_contrib`` tests.
- Radial distribution function analysis
- Periodic many-body expansions with the minimum image convention.
- Many-body alchemical parameter.

### Changed

- Unified use of ``Z``, ``z``, ``R``, and ``r``.
- `dataSet` to `DataSet`
- Switch to qcelemental for atom properties.
- Do not restrict ``sigma_bounds`` for Bayesian optimization after initial grid search.
- ``mbgdml.criteria`` is now ``mbgdml.descriptors`` with a modular handling of descriptors and cutoffs.
- Provide changeable ``Z``, ``R``, ``E``, and ``F`` keys for loading ``npz`` data sets.
- Modularize the error and loss calculations.
- Reduce regularization strength to ``1e-10`` instead of ``1e-15``.
- Split `predict` module into `models` and `predictors`.
- Install bayesian-optimization from git until scipy bug is fixed.
- Merged [sGDML](https://github.com/stefanch/sGDML) changes up to v0.5.4 (``124de3dd8d46a0622bd10c3b4ab033a00dbd3c27``).
- Predict times are logged at debug level.
- Ray must be initialized outside of ``mbePredict`` class.
- Use ``n_workers`` instead of ``n_cores`` in mbePredict.

### Fixed

- ``mbe_worker`` with heterogeneous n-body structures (e.g., solute+solvent).
- Custom ``todict`` method for ASE calculator.
Fixes attached ASE trajectory in reading ``entity_ids``.
- Store ASE Atoms object to avoid recalculating energies and forces in ASE calculator.
- Doc references to respective SchNet functions.

### Removed

- Dependency on ``natsort``.
- ``mbgdml.data.mbModel`` was adsorbed into ``mbgdml.models.gdmlModel``.
- ``structureSets`` and sampling for data sets are no longer supported and subsequently removed.
This functionality was incorporated into [reptar](https://github.com/aalexmmaldonado/reptar).

## [0.0.3] - 2022-07-25

### Added

- [SchNetPack](https://schnetpack.readthedocs.io/en/stable/) prediction capabilities.
- [GAP](https://libatoms.github.io/GAP/index.html) prediction capabilities.
- Training loss function that includes a weighted energy RMSE component.
- Require integration constant evaluation option regardless of performance.
- Initial grid for Bayesian optimization to guide ``sigma_bounds``.
- Ability to keep all trained models instead of just the best one.
- Log parallel optimization.
- Plot Gaussian process from hyperparameter Bayesian optimization.
- Plot cluster losses and population histogram using matplotlib.
- Option to use a sequential reduction optimizer for Bayesian optimization.
- Specify Gaussian process keyword arguments for the final iterative training task.

### Changed

- Removing `md` module in favor of having an `interfaces` module.
- Storage of *n*-body energies and forces in predict sets.
- Redesigned predict methods and parallelized with ray.
- Included a many-body expansion, ``mbe``, module to handle *n*-body energy and force predictions.
- Updated API documentation tree.
- Elements logging in tasks and models are condensed (i.e., no spaces).
- Default ``gp_params`` for Bayesian optimization.
- MD5 hashes are no longer stored in bytes.
- Do not include training set in any problematic clustering.
Training structures are not included in dataset clustering or plots.
- Training JSON to ``training.json`` instead of ``log.json``.
- Iterative training task directory names to state the training set size.

### Fixed

- Added missed torchtools for GDML.
- ``model0`` was not working with iterative training.
- Iterative training would randomly sample every training set.

### Removed

- No longer can make many-body dataset with model predictions (with ``create_mb_from_models``).
- ``e_f_contributions`` was replaced by the ``mbe`` module.

## [0.0.2] - 2022-06-08

### Added

- Iterative training procedure by finding problematic structures.
- Bayesian optimization for hyperparameter search.
- Basic logging capabilities.
- Write JSON file after training with useful information.
- Specify validation structures when training.

### Changed

- Sort ``md5_data`` keys for consistency.
- Renamed ``add_pes_data`` to ``add_pes_json``
- `asdict` is now a method instead of a property.
- Removed sGDML dependency.
- Use relative imports.
- Hyperparameter grid search in ``mbGDMLTrain`` class.
- Moved sGDML modified training routines to ``_train.py``.
- Changed ``Rset_md5`` to ``r_prov_ids`` and ``Rset_info`` to ``r_prov_specs``.
- Improved the ``write_xyz`` and ``string_coords`` functions.
- ``comp_ids`` is now a 1D array where the index of the label is the ``entity_id``.

### Fixed

- Grammar and typos in documentation.
- Address Sphinx documentation warnings and errors.
- Only deploy documentation on keithgroup repo.
- Correct dataSet Rset_info documentation.

### Removed

- ``qc`` module. This does not belong in this package.

## [0.0.1] - 2022-02-24

- Initial release!
