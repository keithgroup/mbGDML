.. _qc-calcs:

==============================
Quantum chemistry calculations
==============================

Once we have a :ref:`data set<data-sets>` populated with structures the next step is to include energies and forces.
These can be manually added, but it is pretty cumbersome and quickly becomes difficult with large data sets.
We offer some ways to streamline quantum chemistry (QC) calculations using external codes and packages.

Slurm calculations
==================

We provide a simple way to create an arbitrary number of Slurm jobs using `Mako <https://www.makotemplates.org/>`__.
The following packages are supported:

- `ORCA <https://orcaforum.kofo.mpg.de/app.php/portal>`__.

With arrays of atomic numbers (``z``) and Cartesian coordinates (``R``), :func:`mbgdml.qc.slurm_engrad_calculation` generates all necessary files to run energy+gradient calculations.

.. automethod:: mbgdml.qc.slurm_engrad_calculation
    :noindex:
