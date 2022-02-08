.. _tut-water-pot:

=========================
Tutorial: Water potential
=========================

We walk through a general framework for developing a mbGDML potential for a single species: water.
At the end, we will have models that predict 1-, 2-, and 3-body interactions at the MP2/def2-TZVP level of theory.

.. note::
    Nothing precludes using a higher, or lower, level of theory.
    Many-body GDML only reproduces the provided energy and forces labels of the data set.
    Recomputing the energies and forces of the data sets is all that is needed to change the level of theory.

Configurational sampling
========================

First, we have to sample across configurational space for *n*-body structures (up to three).
Molecular simulations is a common way to sample structures.
Driving molecular dynamics (MD) simulations with quantum chemistry ensures the most accurate sampling at a high computational cost.
Classical force fields dramatically accelerate these simulations, but parameters are not always available for species of interest.
Semiempirical quantum mechanics (SQM) methods offer a compromise of speed and generalizability to most chemical species of interest.
For this tutorial, we will use the `GFN2-xTB <https://doi.org/10.1021/acs.jctc.8b01176>`_ method to drive a MD simulation of a water droplet using the `xtb program <https://xtb-docs.readthedocs.io/en/latest/contents.html>`_.
Sampling at a higher temperature, say 500 K, helps 

`Packmol <http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml>`_ is used to generate an initial structure for the simulation.
Since xtb `xtb program <https://xtb-docs.readthedocs.io/en/latest/contents.html>`_ only has spherical confining potentials, we will generate a spherical droplet of water.
The Cartesian coordinates of a water monomer is needed for `Packmol <http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml>`_.
Here is the `MP2/def2-TZVP optimized geometry of a water molecule <https://github.com/keithgroup/solute-solvent-clusters/tree/main/clusters/homogeneous/h2o/1h2o/1h2o.abc>`_ that we use called ``1h2o.abc.mp2.def2tzvp.xyz``.

.. code-block:: text
    
    3

    O      0.000000    0.000000    0.216072
    H      0.000000    0.761480   -0.372151
    H      0.000000   -0.761480   -0.372151

Now we can use Packmol to generate a sphere.


.. code-block:: text

    tolerance 2.0
    output {molecules}{solvent_label}.sphere-packmol.xyz
    filetype xyz
    structure {monomer_path}
        number {molecules}
        inside sphere 0.0 0.0 0.0 {float(radius)}
    end structure
