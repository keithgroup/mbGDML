<h1 align="center">mbGDML</h1>

<h4 align="center">Create, use, and analyze many-body, gradient-domain machine learning potentials.</h4>

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
</p>

<p align="center">
    <a href="#about">About</a> •
    <a href="#features">Features</a>  •
    <a href="#installation">Installation</a>  •
    <a href="#license">License</a>
</p>

# About

Atomistic insight is fundamental for computational predictive studies of chemical and physical processes.
Machine learning potentials (or force fields) can reproduce high-level ab initio calculations without the substantial costs.
[Gradient-domain machine learning (GDML)](http://quantum-machine.org/gdml/), a kernel-based method, directly learns the relationship between atomic coordinates and interatomic forces with only hundreds of data points.
However, it uses the inverse internuclear distance descriptor which sacrifices generalized transferability to other species or number of atoms.

[Many-body GDML (mbGDML)](https://github.com/keithgroup/mbGDML), is a route for GDML size-transferable potentials by using GDML to learn *n*-body interactions for use in a many-body expansion approach.
Every aspect of the process from preparing energy+gradient calculations, creating data sets, training GDML models, and making predictions is taken care of in this user-friendly Python package.

**Disclaimer**: This package is still under active development and is not suitable for production.

# Features

Creation:

- Partition structures into monomers, dimers, trimers, etc.
- Structure and data set breadcrumb trails using MD5 hashes.
- Simple GDML training interface using CPUs.

Use:

- Energy and force predictions parallelized with [ray](https://docs.ray.io/en/latest/).
- Periodic boundary conditions with the minimum image convention.
- Calculator interface with the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/).

Analysis:

- Avoid recalculating energies and forces by storing predictions into predict sets (npz files).
- Visually examine structural similarity and prediction accuracy using dimensionality reduction with [UMAP](https://umap-learn.readthedocs.io/en/latest/).
- Analyze *n*-body contributions and predictions with heat maps.

Other supported models:

- [SchNetPack](https://schnetpack.readthedocs.io/en/stable/)
- [GAP](https://libatoms.github.io/GAP/)

# Installation

You can install mbGDML by using `pip install mbgdml`.
Or, the latest development version can be installed directly from the [GitHub repository](https://github.com/keithgroup/mbGDML).

```text
git clone https://github.com/keithgroup/mbGDML
cd mbGDML
pip install .
```

# License

Distributed under the MIT License. See `LICENSE` for more information.
