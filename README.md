<h1 align="center">mbGDML</h1>

<h4 align="center">Create, use, and analyze machine learning potentials within the many-body expansion framework.</h4>

<h4 align="center" style="padding-bottom: 0.5em;"><a href="https://keithgroup.github.io/mbGDML">Documentation</a></h4>

<p align="center">
    <a href="https://github.com/keithgroup/mbGDML/actions/workflows/python-package.yml">
        <img src="https://github.com/keithgroup/mbGDML/actions/workflows/python-package.yml/badge.svg" alt="Build Status ">
    </a>
    <a href="https://codecov.io/gh/keithgroup/mbGDML">
        <img src="https://codecov.io/gh/keithgroup/mbGDML/branch/main/graph/badge.svg?token=P643JEUWZC" alt="Codecov">
    </a>
    <a href="https://doi.org/10.5281/zenodo.6270373">
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6270373.svg" alt="DOI">
    </a>
    <a href="https://github.com/keithgroup/mbGDML/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/github/license/keithgroup/mbGDML" alt="License">
    </a>
    <a href="https://github.com/keithgroup/mbGDML" target="_blank">
        <img src="https://img.shields.io/github/repo-size/keithgroup/mbGDML" alt="Repo size">
    </a>
    <a href="https://github.com/psf/black" target="_blank">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black style">
    </a>
    <a href="https://github.com/PyCQA/pylint" target="_blank">
        <img src="https://img.shields.io/badge/linting-pylint-yellowgreen" alt="Black style">
    </a>
</p>

<p align="center">
    <a href="#motivation">Motivation</a> •
    <a href="#approach">Approach</a> •
    <a href="#features">Features</a>  •
    <a href="#installation">Installation</a>  •
    <a href="#license">License</a>
</p>

## Motivation

Machine learning potentials (i.e., force fields) often rely on local descriptors for size transferability.
These descriptors partition total properties into atomic contributions; however, they inherently neglect complicated long-range interactions by enforcing atomic radial cutoffs.
Global descriptors encode the entire structure with no cutoffs and can capture interactions at all scales.
However, they are restricted to systems with the same number of atoms.

<img src="https://github.com/keithgroup/mbGDML/blob/main/docs/source/images/descriptors/global-vs-local-descriptor.png?raw=true" width="600"/>

[Gradient-domain machine learning](http://www.sgdml.org/) (GDML) is one example of a ML potential with a global descriptor.
GDML is unique because it trains directly on forces and recovers total energy through analytical integration.
This provides substantially more information about the potential energy surface (PES) and allows for better interpolation between training data.
As a result, GDML typically only needs 1000 structures to accurately learn energies and forces.

<img src="https://github.com/keithgroup/mbGDML/blob/main/docs/source/images/gdml-concept-e-vs-f-train.png?raw=true" width="500"/>

To date, GDML has been limited to the exact system it was trained on.
This makes simulations on arbitrarily size systems, like solvents, futile.

## Approach

Many-body expansions (MBEs) rigorously decomposes total (i.e., supersystem) energies into fundamental *n*-body interactions.
This expansion is formally exact when all *N*-body interactions are accounted for.
In practice, however, it is typically truncated to the third order.
One can then model any system by summing up 1-, 2-, and 3-body contributions.

<img src="https://github.com/keithgroup/mbGDML/blob/main/docs/source/images/explicit-water-methanol-mbe-allorders.svg?raw=true" width="400"/>

MBEs driven by GDML potentials trained on *n*-body interactions is a promising approach for size-transferable potentials.
Furthermore, GDML model's remarkable data efficiency enables training on highly accurate quantum chemical methods.

## Features

### Train

- Train GDML models using grid searches, Bayesian optimization, or both on CPUs.
- Custom loss functions.
- Iterative training procedure for automated curation of optimal training sets.

### Predict

- Many-body predictions with GDML, [SchNet](https://schnetpack.readthedocs.io/en/stable/) and [GAP](https://libatoms.github.io/GAP/) potentials.
- Parallel GDML predictions with [ray](https://docs.ray.io/en/latest/) from a laptop to multiple nodes.
- Periodic structures with the minimum-image convention.
- Alchemical predictions by tuning out 2- or 3-body contributions of specific entities.

### Analysis

- Prediction sets that store decomposed predictions for further analysis.
- Radial distribution functions.
- Cluster and identify problematic (i.e., high error) structures using [sklearn](https://scikit-learn.org/stable/index.html).

### Interfaces

- [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/) (ASE) for geometry optimizations, molecular dynamics simulations, and more.

## Installation

You can install mbGDML from [PyPI](https://pypi.org/project/mbGDML/) by using `pip install mbGDML`.
Or, the latest development version can be installed directly from the [GitHub repository](https://github.com/keithgroup/mbGDML) or from [TestPyPI](https://test.pypi.org/project/mbGDML/).

```text
git clone https://github.com/keithgroup/mbGDML
cd mbGDML
pip install .
```

## License

Distributed under the MIT License. See `LICENSE` for more information.
