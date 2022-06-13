# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Ability to keep all trained models instead of just the best one.
- Log parallel optimization.
- Plot Gaussian process from hyperparameter Bayesian optimization.
- Plot cluster losses and population histogram using matplotlib.
- Option to use a sequential reduction optimizer for Bayesian optimization.
- Specify Gaussian process keyword arguments for the final iterative training task.

### Changed

- Elements logging in tasks and models are condensed (i.e., no spaces).
- Default ``gp_params`` for Bayesian optimization.
- MD5 hashes are no longer stored in bytes.
- Do not include training set in any problematic clustering.
Training structures are not included in dataset clustering or plots.
- Training JSON to ``training.json`` instead of ``log.json``.
- Iterative training task directory names to state the training set size.

### Fixed

- ``model0`` was not working with iterative training.
- Iterative training would randomly sample every training set.

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
