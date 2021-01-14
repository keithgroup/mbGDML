

<h1 align="center">mbGDML</h1>

<h4 align="center">Create, use, and analyze many-body gradient domain machine learning force fields.</h4>

<p align="center">
    <a href="https://travis-ci.com/keithgroup/mbGDML">
        <img src="https://travis-ci.com/keithgroup/mbGDML.svg?branch=master" alt="Build Status ">
    </a>
    <a href="https://codecov.io/gh/keithgroup/mbGDML">
        <img src="https://codecov.io/gh/keithgroup/mbGDML/branch/master/graph/badge.svg?token=P643JEUWZC" alt="Codecov">
    </a>
    <a href="https://lgtm.com/projects/g/keithgroup/mbGDML/context:python">
        <img src="https://img.shields.io/lgtm/grade/python/g/keithgroup/mbGDML.svg?logo=lgtm&logoWidth=18" alt="Language grade: Python">
    </a>
    <a href="https://github.com/keithgroup/mbGDML/blob/master/LICENSE" target="_blank">
        <img src="https://img.shields.io/github/license/keithgroup/mbGDML" alt="License">
    </a>
</p>

<p align="center">
    <a href="#about">About</a> •
    <a href="#features">Features</a>  •
    <a href="#installation">Installation</a>
</p>


# About

Atomistic insight is fundamental for computational predictive studies of chemical and physical processes.
Machine learning force fields provide a route to high-level ab initio calculations at a fraction of the cost.
[Symmetric gradient domain machine learning (sGDML)](http://quantum-machine.org/gdml/), a kernel-based method, directly learns the relationship between atomic coordinates and interatomic forces.
However, training in the gradient domain sacrifices generalized transferability to other species or number of atoms.

[Many-body GDML (mbGDML)](https://github.com/keithgroup/mbGDML), is a technique for sGDML transferability to *n*-sized systems by using many-body machine learning models.
Every aspect of the process from preparing ORCA calculations, data set creation, training, and use of mbGDML force fields is taken care of in this user-friendly Python package.

**Disclaimer**: This package is still under active development and is not ready for production.

# Features

Creating mbGDML models:

* Partition structures into monomers, dimers, trimers, etc.
* Prepare and submit [ORCA](https://sites.google.com/site/orcainputlibrary/) energy and gradient calculations.
* Conversion between data sets ([NumPy npz](https://numpy.org/doc/stable/reference/routines.io.html) files) and output files.
* Train sGDML models.

Using mbGDML models:

* Energy and force prediction of structures with mbGDML models.
* Interface with the [atomic simulation environment (ASE)](https://wiki.fysik.dtu.dk/ase/).

Analyzing mbGDML models:

* Store mbGDML predictions into predict data sets.
* Analyze n-body contributions and create heat maps.

# Installation

At the moment, the only way to install mbGDML is directly from the [GitHub repository](https://github.com/keithgroup/mbGDML).

## Requirements

The following packages are required:

* ase
* cclib (>=1.7)
* dscribe
* mako
* matplotlib
* natsort
* numpy
* sgdml

All of these required packages can be installed with:

```
pip install ase 'cclib>=1.7' dscribe mako matplotlib natsort numpy sgdml
```

Then, clone and install the repository.

```
git clone https://github.com/keithgroup/mbGDML
cd mbGDML
pip install .
```