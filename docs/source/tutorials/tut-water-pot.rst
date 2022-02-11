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
We will use a higher temperature, say 500 K, to help sample low- and high-energy configurations.

`Packmol <http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml>`_ is used to generate an initial structure for the simulation.
Since the `xtb program <https://xtb-docs.readthedocs.io/en/latest/contents.html>`_ only has spherical confining potentials, we will generate a spherical droplet of water.
The Cartesian coordinates of a water monomer is needed for `Packmol <http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml>`_.
Here is the `MP2/def2-TZVP optimized geometry of a water molecule <https://github.com/keithgroup/solute-solvent-clusters/tree/main/clusters/homogeneous/h2o/1h2o/1h2o.abc>`_ that we use called ``1h2o.abc.mp2.def2tzvp.xyz``.

.. code-block:: text
    
    3

    O      0.000000    0.000000    0.216072
    H      0.000000    0.761480   -0.372151
    H      0.000000   -0.761480   -0.372151

.. warning::

    GDML models use global descriptors (specifically inverse internuclear distances).
    This means the order of the atoms matters, and every structure we want to predict has to provide the Cartesian coordinates in this order: the Oxygen atom, then the hydrogens.
    Because of symmetry we can specify the hydrogen atoms in any order.

    In a molecule such as methanol the methyl hydrogens are distinguishable---with respect to the hydroxyl group---and their order matters with respect to the hydroxyl group and rotation (i.e., clockwise or counterclockwise).
    However, just the rotational order of the hydrogens on acetonitrile would matter.

    Thus, setting the order of the atoms here, in the monomer structure, is highly recommended.

Now we can use `packmol <http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml>`_ to generate a sphere with a diameter of, say, 20 Angstroms.
This size would ensure we can sample enough long-distance structures.
Using properties of the molecule (i.e., mass density and molar mass) we can compute the number of molecules to include for a 500 K droplet of water---this ends up being around 140 water molecules.
A `packmol <http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml>`_ input file to generate this structure is shown below.
The resulting structure will look something like :download:`this xyz file<../files/tut-water/140h2o.pm.xyz>`.

.. code-block:: text

    tolerance 2.0
    output 140h2o.sphere-packmol.xyz
    filetype xyz
    structure 1h2o.abc.mp2.def2tzvp.xyz
        number 140
        inside sphere 0.0 0.0 0.0 10.0
    end structure

Now we are ready to prepare a GFN2-xTB simulation.
Minimizing the `packmol <http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml>`_ structure ensures we have a reasonable starting structure.
Since we are using GFN2-xTB for the MD simulation, we first need to optimize it using the same level of theory.
We can do this with the `xtb program <https://xtb-docs.readthedocs.io/en/latest/contents.html>`_ using the following command.
We will get something like :download:`the following structure<../files/tut-water/140h2o.pm.gfn2.xyz>`.

.. code-block:: bash

    xtb ./140h2o.pm.xyz --opt tight --gfn 2 --charge 0 --cycles 200

Now we are ready to run a GFN2-xTB simulation using the input file below named ``140h2o.pm.gfn2-xtb.md.eq1-gfn2.500k.wallpot.inp``.
Basically, this runs a 10 picosecond NVT simulation at 500 K with a 1 femtosecond integration time step to equilibrate the system.
A spherical confining potential is used to prevent dissociation during the run.

.. code-block:: text

    $md
        restart  = false
        time     =  10.0  # in ps (1000 fs = 1 ps; 1 fs = 0.001 ps)
        step     =   1.0  # in fs
        dump     =  10.0  # in fs
        temp     = 500.0  # in K
        nvt      = true
        velo     = false
        hmass    =   0  # Use the standard hydrogen mass.
        shake    =   0
        sccacc   =   1.0
    $end

    $wall
        potential = logfermi
        sphere: auto, all
        temp = 300.0
        beta = 6
    $end

    $write
        wiberg=false
        dipole=false
        charges=false
        mulliken=false
        orbital energies=false
        inertia=false
        distances=false
        final struct=false
        geosum=false
    $end

With the optimized xyz structure, we start the simulation using the following command.

.. code-block:: bash

    xtb ./140h2o.pm.gfn2.xyz --md --input 140h2o.pm.gfn2-xtb.md.eq1-gfn2.500k.wallpot.inp --gfn 2 --charge 0 --verbose

After confirming the system is fully equilibrated we will run a production simulation.
We mostly reuse the previous output file and command with two changes.

- **Time of the simulation.**
  Instead we can run a simulation for 1 picosecond.
- **Manually specify the confining potential size.**
  Before, we had a line in the ``$wall`` block that states ``sphere: auto, all`` which automatically determines the radius of the sphere and applies it to all atoms.
  The size of the confining potential could change when restarting the simulation.
  Thus, manually specifying the radius, in Bohr, is recommended.
  The output file of the previous simulation will have a line that says something like ``spherical wallpotential with radius   11.5624559 Å``.
  We just have to convert this to Bohr, which is about ``21.84987498332`` and specify it like so: ``sphere: 21.84987498332, all``.

With the production simulation coordinates in hand we can being preparing a structure set.

Creating a structure set
========================
