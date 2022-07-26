.. _data-sets:

=========
Data sets
=========

The foundation of ML methods is the data set.
It represents the collection of structures, energies, and forces that GDML trains on and potential energy surface it reproduces.
Structures are always ``sampled`` from one or more :ref:`structure sets<structure-sets>`.


.. _load-save-dset:

Loading and saving
------------------

Data sets are stored as NumPy ``.npz`` files.
To load a data set, you can pass the path to a ``dataset.npz`` file or explicitly use the :meth:`~mbgdml.data.dataSet.load` function.

.. code-block:: python

    from mbgdml.data import dataSet

    dset = dataSet('./path/to/dataset.npz')
    # Or
    dset = dataSet()
    dset.load('./path/to/dataset.npz')


Saving a data set can be done using the :meth:`~mbgdml.data.dataSet.save` function.
The required ``data`` dictionary for ``save`` is provided as the ``asdict`` attribute which creates a dictionary of all data to save in the ``npz`` file.

.. code-block:: python

    dset.save('dataset', dset.dataset, './path/to')




Creation
--------

Data sets are created in two stages: structural sampling and calculating energy and forces.

Structure sampling
^^^^^^^^^^^^^^^^^^

Curating a data set starts by sampling geometries from :ref:`structure sets<structure-sets>` or even data sets with :meth:`~mbgdml.data.dataSet.sample_structures`.

Energies and forces
^^^^^^^^^^^^^^^^^^^

These data need to be computed using your program and then stored in the :obj:`mbgdml.data.dataSet` object using the :attr:`~mbgdml.data.dataSet.E` and :attr:`~mbgdml.data.dataSet.F` attributes.
Units for energies and forces are normally kcal/mol and kcal/(mol A) for mbGDML models.

Unit conversion
---------------

We provide a simple way to convert Cartesian coordinates, energies, or forces to a variety of units.

- :meth:`~mbgdml.data.dataSet.convertR`
- :meth:`~mbgdml.data.dataSet.convertE`
- :meth:`~mbgdml.data.dataSet.convertF`

So, say we wanted to convert ``dset`` energies and forces from hartree and hartree/A to kcal/mol and kcal/(mol A).
The coordinates are already in Angstroms, so we just need to convert the energies and forces.

.. code-block:: python
    
    my_dataset.convertE('kcal/mol')
    my_dataset.convertF('hartree', 'bohr', 'kcal/mol', 'Angstrom')

.. warning::

    ``convertF`` does not change any unit specifications (i.e., ``r_unit`` and ``e_unit``), but **needs** to match both coordinate and energy units.




.. _mb-data-sets:

Many-body data
--------------

GDML models for the many-body expansion require energies and forces where the lower-order contributions (i.e., 1-body) are removed (i.e., dimers).
We indicate that a data set contains many-body data with the following attributes.

.. autoattribute:: mbgdml.data.dataSet.mb
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.mb_dsets_md5
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.mb_models_md5
    :noindex:

Removing *n*-body contributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :meth:`~mbgdml.data.dataSet.create_mb_from_dsets`
    

Contents
--------

Data set objects contain the following information .

.. autoattribute:: mbgdml.data.dataSet.name
    :noindex:

Atomic numbers
^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.dataSet.z
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.n_z
    :noindex:

Cartesian Coordinates
^^^^^^^^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.dataSet.R
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.n_R
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.r_unit
    :noindex:

MD5 hash
^^^^^^^^

.. autoattribute:: mbgdml.data.dataSet.md5
    :noindex:

.. note::
   MD5 hashes are recomputed whenever the attribute is called.
   Thus, the MD5 hash will be irreversibly changed if any changes are made to MD5-hash relevant data. 

Structure set identification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.dataSet.r_prov_ids
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.r_prov_specs
    :noindex:

Structure sampling
^^^^^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.dataSet.criteria
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.z_slice
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.cutoff
    :noindex:

Structure properties
^^^^^^^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.dataSet.E
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.e_unit
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.E_mean
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.E_min
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.E_max
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.F
    :noindex:

.. note::
    There is no explicit force unit attribute in a data set.
    It is assumed to be ``e_unit`` ``r_unit``:sup:`-1`.

.. autoattribute:: mbgdml.data.dataSet.F_mean
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.F_min
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.F_max
    :noindex:

.. autoattribute:: mbgdml.data.dataSet.theory
    :noindex:
