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
        dump     =  50.0  # in fs
        temp     = 500.0  # in K
        nvt      = true
        velo     = false
        hmass    =   1
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
  
  .. note::
      One could also have specified a radius corresponding to the initial `packmol <http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml>`_ shape instead of letting `xtb <https://xtb-docs.readthedocs.io/en/latest/contents.html>`_ automatically determine it.
      The density of system usually significantly changes when the ``auto`` option is used.

With the :download:`production simulation trajectory<../files/tut-water/140h2o.pm.gfn2.md.500k.eq0-xtb.md.prod1-gfn2.500k.wallpot.xyz>` in hand we can being preparing a structure set.

Creating structure sets
=======================

Sometimes data sets contain information from a variety of sources and we, as practitioners of reproducible research, need to keep a breadcrumb trail of our data.
:ref:`Structure sets<structure-sets>` allow us to create a collection of structures derived from the same source (e.g., a MD simulation, global optimization, or article) along with a unique :attr:`~mbgdml.data.structureset.structureSet.md5` identifier.
Fragment/molecule specification is also defined in this stage that lets mbGDML correctly identify which model to use for each fragment.
All we need to start is a single XYZ file (our GFN2-xTB trajectory will serve this purpose).

Besides the XYZ file, only three other pieces of information are required: :attr:`~mbgdml.data.structureset.structureSet.r_unit`, :attr:`~mbgdml.data.structureset.structureSet.entity_ids`, :attr:`~mbgdml.data.structureset.structureSet.comp_ids`.
For small systems you can manually generate the :attr:`~mbgdml.data.structureset.structureSet.entity_ids` and :attr:`~mbgdml.data.structureset.structureSet.comp_ids` manually.
Two water molecules would just be ``[0, 0, 0, 1, 1, 1]`` and ``[['0', 'h2o'], ['1', 'h2o']``, respectively.

.. note::
    Any label can be used for the component id.
    For simplicity we will just use ``h2o``.

Larger systems become more tedious to manually prepare.
We can use :func:`mbgdml.utils.get_entity_ids` and :func:`mbgdml.utils.get_comp_ids` to automatically generate :attr:`~mbgdml.data.structureset.structureSet.entity_ids` and :attr:`~mbgdml.data.structureset.structureSet.comp_ids` for systems containing only one species (e.g., all water molecules).
The following code will generate a :ref:`structure set<structure-sets>` just like :download:`this one<../files/tut-water/140h2o.pm.gfn2.md.500k.prod1.npz>`.

.. code-block:: python

    from mbgdml.data import structureSet
    from mbgdml.utils import get_entity_ids, get_comp_ids

    # Path to xyz file we will turn into a structure set.
    xyz_path = './140h2o.pm.gfn2.md.500k.eq0-xtb.md.prod1-gfn2.500k.wallpot.xyz'

    name = '140h2o.pm.gfn2.md.500k.prod1'
    r_unit = 'Angstrom'  # Coordinate units.
    comp_id = 'h2o'  # Component ID.
    atoms_per_entity = 3  # Number of atoms in each entity.
    num_entities = 140  # Number of entities in each XYZ structure.

    entity_ids = get_entity_ids(atoms_per_entity, num_entities)
    comp_ids = get_comp_ids(comp_id, entity_ids)

    rset = structureSet()
    rset.from_xyz(xyz_path, r_unit, entity_ids, comp_ids)  # Adds data to structure set.
    rset.name = name  # Assigns name to the structure set.
    rset.save(rset.name, rset.asdict)  # Will save in current directory.

Curating data sets
==================

Now we can start building a :ref:`data set<data-sets>` of *n*-body structures containing energies and forces.
This generally comes in two stages: sampling structures from :ref:`structure set<structure-sets>` or other :ref:`data set<data-sets>` then computing energies and forces using some quantum chemical method.

Sampling structures
-------------------

A structurally diverse data set is paramount for globally accurate (i.e., useful) GDML models.
We sample structures from any valid :ref:`structure set<structure-sets>` multiple times as the structures are just appended to the end of the arrays.

Remember that the final :ref:`data sets<data-sets>` will need to contain *n*-body energies and forces.
Meaning once we will need properties for each clusters' lower order (< *n*) fragments.
For example, a dimer's two-body energy is defined as the total energy minus the energies of the individual monomers.
What we are getting at is that many-body :ref:`data sets<data-sets>` require more calculations than just its own structures.

In order to minimize the number of calculations required we can build the lower order :ref:`data sets<data-sets>` (e.g., 1- and 2-body in this case) only from the highest-order :ref:`data set<data-sets>` (e.g., 3-body).
Thus, we only directly sample from the MD :ref:`structure set<structure-sets>` once to make the 3-body :ref:`data set<data-sets>`.
1- and 2-body :ref:`data sets<data-sets>` are then sampled from the 3-body set instead.

.. note::
    This procedure does bias the 1- and 2-body data sets to only contain structures from the sampled 3-body data set.
    Nothing precludes additional 1- and 2-body sampling on top of the required structures.
    We do not do any additional sampling just to keep the number of calculations small.

Distance-based cutoffs
~~~~~~~~~~~~~~~~~~~~~~

Many-body expansions suffer from the curse of dimensionality.
The number of *n*-body clusters explodes when the overall size of the system increases.
Luckily, the size of the *n*-body contribution generally decreases when the distance between the individual molecules increases.
If we employ some distance-based cutoff we can avoid computations on structures with negligible contributions.

One common distance-based cutoff is what we call the "center of mass distance sum".
Essentially we take the sum of each monomer's center of mass to the center of mass of the entire cluster.
Example distances for a water trimer are shown below.

.. image:: ../images/distance-screening-3h2o.svg
   :width: 300px
   :align: center

We provide a simple function :func:`mbgdml.criteria.cm_distance_sum` can compute this metric.
Different criteria can be used or added, but so far each one requires four parameters:

* ``z``: atomic numbers of the structure;
* ``R``: Cartesian coordinates;
* ``z_slice``: relevant atom indices for the criteria;
* ``entity_ids``: entity IDs for every atom in the structure.

.. note::
    Not every criteria will always use ``z_slice`` or ``entity_ids``.
    We just always require them to standardize their use in scripts.
    If a criteria does not use a parameter then just pass a "None" equivalent.
    For example, :func:`mbgdml.criteria.cm_distance_sum` does not use ``z_slice`` directly in the criteria evaluation so we just pass ``np.array([])`` to the function.

To determine cutoffs we typically calculate all *n*-body interactions for a large structure.
We then plot the predicted *n*-body energy when using different cutoffs and determine when it reasonably converges.
For water we will use 6 and 10 Angstroms for the 2- and 3- body models.

Sampling from structure sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following Python script will sample 5,000 trimer structures from our production MD simulation and result in a :download:`data set without energies and forces<../files/tut-water/140h2o.pm.gfn2.md.500k.prod1.3h2o-dset-cm10.noef.npz>`.

.. code-block:: python

    import os
    import numpy as np
    from mbgdml.data import structureSet, dataSet
    from mbgdml.criteria import cm_distance_sum

    rset_path = './140h2o.pm.gfn2.md.500k.prod1.npz'
    dset_name = '140h2o.pm.gfn2.md.500k.prod1.3h2o-dset-cm10.noef'
    save_dir = '.'

    # How many monomers to include in each sampled structure?
    size = 3
    # How many structures to sample?
    # A number (e.g., `5000`) or `'all'`.
    quantity = 5000
    # Only accept structures that pass some criteria.
    r_criteria = cm_distance_sum  # None for 1mer, cm_distance_sum for others.
    z_slice = np.array([])  # Specifies which atoms to use for a criteria.
    cutoff = np.array([10.0])  # Angstroms; [] for 1mer, [##] for others.
    # Will translate the center of mass of the sampled cluster to the origin.
    center_structures = True
    # Will print sampling updates.
    sampling_updates = True

    # Ensures we execute from script directory (for relative paths).
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    if save_dir[-1] != '/':
        save_dir += '/'

    # Preparing data set.
    dset = dataSet()
    dset.name = dset_name

    # Sampling from structure set.
    rset = structureSet(rset_path)
    dset.sample_structures(
        rset, quantity, size, criteria=r_criteria, z_slice=z_slice,
        cutoff=cutoff, center_structures=center_structures,
        sampling_updates=sampling_updates
    )

    dset.save(dset.name, dset.asdict, save_dir)

Sampling from data sets
~~~~~~~~~~~~~~~~~~~~~~~

Once we have our trimer data set in hand we can sample all 1- and 2-body structures available.
The script is very similar with only a few modifications.

.. code-block:: python

    import os
    import numpy as np
    from mbgdml.data import dataSet
    from mbgdml.criteria import cm_distance_sum

    dset_path_for_sampling = './140h2o.pm.gfn2.md.500k.prod1.3h2o-dset-cm10.noef.npz'
    dset_name = '140h2o.pm.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o-dset-noef'
    save_dir = '.'

    # How many monomers to include in each sampled structure?
    size = 2
    # How many structures to sample?
    # A number (e.g., `5000`) or `'all'`.
    quantity = 'all'
    # Only accept structures that pass some criteria.
    r_criteria = None  # None for 1mer or dset sampling, cm_distance_sum for others.
    z_slice = np.array([])  # Specifies which atoms to use for a criteria.
    cutoff = np.array([])  # Angstroms; [] for 1mer, [##] for others.
    # Will translate the center of mass of the sampled cluster to the origin.
    center_structures = True
    # Will print sampling updates.
    sampling_updates = True

    # Ensures we execute from script directory (for relative paths).
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    if save_dir[-1] != '/':
        save_dir += '/'

    # Preparing data set.
    dset = dataSet()
    dset.name = dset_name

    # Sampling from data set.
    dset_for_sampling = dataSet(dset_path_for_sampling)
    dset.sample_structures(
        dset_for_sampling, quantity, size, criteria=r_criteria, z_slice=z_slice,
        cutoff=cutoff, center_structures=center_structures,
        sampling_updates=sampling_updates
    )

    dset.save(dset.name, dset.asdict, save_dir)

The above script directly results in :download:`this dimer data set<../files/tut-water/140h2o.pm.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o-dset-noef.npz>`.
By changing ``dset_name`` and ``size = 2`` to ``1`` we get :download:`this monomer data set<../files/tut-water/140h2o.pm.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-dset-noef.npz>`.

Computing energies and forces
-----------------------------

TODO: Add information.

.. code-block:: python

    import os
    import numpy as np
    from mbgdml.data import dataSet
    from mbgdml.qc import slurm_engrad_calculation

    # Script setup.
    dset_path = '140h2o.pm.gfn2.md.500k.prod1.3h2o-dset-cm10.noef.npz'  # The data set to make engrad jobs from.
    structure_label = '140h2o.pm.gfn2.md.500k.prod1.3h2o.cm10'  # Structure label for the engrad calculations.
    calc_name = f'{structure_label}-orca.engrad-mp2.def2tzvp'  # Job name.
    max_calcs = 100  # Maximum number of consecutive calculations to put in one job.

    # Ensures we execute from script directory (for relative paths).
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    save_dir = f'./{structure_label}-calcs'
    if save_dir[-1] != '/':
        save_dir += '/'
    os.makedirs(save_dir, exist_ok=True)

    def prepare_calc(calc_name, z, R, save_dir):
        slurm_engrad_calculation(
            'orca',
            z,
            R,
            calc_name,
            calc_name,
            calc_name,
            theory='MP2',
            basis_set='def2-TZVP',
            charge=0,
            multiplicity=1,
            cluster='smp',
            nodes=1,
            cores=24,
            days=3,
            hours=00,
            calc_dir=save_dir,
            options='TightSCF FrozenCore',
            control_blocks=(
                '%maxcore 8000\n\n'
                '%scf\n    ConvForced true\nend'
            ),
            write=True,
            submit=False
        )

    dset = dataSet(dset_path)
    missing_engrad_indices = np.argwhere(np.isnan(dset.E))[:,0]

    # Splits up calculations to a maximum number of engrads per job.
    if len(missing_engrad_indices) > max_calcs:
        start = 0
        end = max_calcs
        while start < len(missing_engrad_indices):
            if end > len(missing_engrad_indices):
                end = len(missing_engrad_indices)
            calc_name_iter = f'{calc_name}-{start}.to.{end-1}'
            save_dir_calc = f'{save_dir}/{calc_name_iter}'

            if save_dir_calc[-1] != '/':
                save_dir_calc += '/'
            
            os.makedirs(save_dir_calc, exist_ok=True)

            prepare_calc(calc_name_iter, dset.z, dset.R[start:end], save_dir_calc)

            start += max_calcs
            end += max_calcs
    else:
        prepare_calc(calc_name, dset.z, dset.R[missing_engrad_indices], save_dir)

A bunch of ORCA 4.2.0 jobs will be generated from the above script.
Each job will contain up to 100 energy+gradient calculations with the specified parameters.
The first two calculations are shown below.

.. code-block:: text

    # 140h2o.pm.gfn2.md.500k.prod1.3h2o.cm10-orca.engrad-mp2.def2tzvp-0.to.99
    ! MP2 def2-TZVP EnGrad TightSCF FrozenCore

    %pal
        nprocs 24
    end

    %maxcore 8000

    %scf
        ConvForced true
    end

    *xyz 0 1
    O   0.577096335  -2.910013209  -1.469728926
    H   0.719154865  -2.645061494  -2.428305146
    H   1.514695271  -3.029077780  -1.202822563
    O  -0.046882667   0.333722805  -0.023751790
    H   0.547109024   0.746187830  -0.616407617
    H  -0.118653895   0.901757409   0.740811852
    O  -0.648361732   2.498671995   1.470652220
    H   0.177319851   3.012341517   1.445782624
    H  -0.964075000   2.246011633   2.423333479
    *


    $new_job

    # 140h2o.pm.gfn2.md.500k.prod1.3h2o.cm10-orca.engrad-mp2.def2tzvp-0.to.99
    ! MP2 def2-TZVP EnGrad TightSCF FrozenCore

    %pal
        nprocs 24
    end

    %maxcore 8000

    %scf
        ConvForced true
    end

    *xyz 0 1
    O   2.186900029   1.103131746   0.440180559
    H   3.132328705   1.052402431   0.150975525
    H   1.713400705   2.019828029   0.343004762
    O  -2.135618996  -1.688550022   0.713837639
    H  -2.773132665  -0.989875555   0.770465470
    H  -1.826474091  -1.487707395  -0.182951687
    O  -0.064924338   0.527011920  -1.137617882
    H  -0.007597048   0.789579633  -0.215653695
    H  -0.021943960  -0.457051162  -1.126188389
    *



Adding energies and forces
--------------------------

TODO: Add qcjson information.

:func:`~mbgdml.data.dataset.dataSet.add_pes_data`

:download:`data set with energies and forces<../files/tut-water/140h2o.pm.gfn2.md.500k.prod1.3h2o-dset-cm10.npz>`

Many-body data sets
-------------------

TODO

:func:`~mbgdml.data.dataset.dataSet.create_mb_from_dsets`

Training GDML models
====================

TODO

:func:`~mbgdml.train.mbGDMLTrain.train`

Making predictions
==================

TODO

:func:`~mbgdml.predict.mbPredict.predict`

