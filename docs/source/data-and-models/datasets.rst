.. _data-sets:

=========
Data sets
=========

The foundation of ML methods is the data set.
It represents the collection of structures, energies, and forces that GDML trains on and potential energy surface it reproduces.
Structures are always ``sampled`` from one or more :ref:`structure sets<structure-sets>`.






Contents
--------

Data sets contain the following information.

.. note::
    Some data and information pertaining to :ref:`many-body data sets<mb-data-sets>` are discussed later.

Atomic numbers
^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.dataset.dataSet.z

.. autoattribute:: mbgdml.data.dataset.dataSet.n_z

Cartesian Coordinates
^^^^^^^^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.dataset.dataSet.R

.. autoattribute:: mbgdml.data.dataset.dataSet.n_R

.. autoattribute:: mbgdml.data.dataset.dataSet.r_unit

MD5 hash
^^^^^^^^

.. autoattribute:: mbgdml.data.dataset.dataSet.md5

.. note::
   MD5 hashes are recomputed whenever the attribute is called.
   Thus, the MD5 hash will be irreversibly changed if any changes are made to MD5-hash relevant data. 

Structure set identification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.dataset.dataSet.Rset_md5

.. autoattribute:: mbgdml.data.dataset.dataSet.Rset_info

Structure sampling
^^^^^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.dataset.dataSet.criteria

.. autoattribute:: mbgdml.data.dataset.dataSet.z_slice

.. autoattribute:: mbgdml.data.dataset.dataSet.cutoff

Structure properties
^^^^^^^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.dataset.dataSet.E

.. autoattribute:: mbgdml.data.dataset.dataSet.e_unit

.. autoattribute:: mbgdml.data.dataset.dataSet.E_mean

.. autoattribute:: mbgdml.data.dataset.dataSet.E_min

.. autoattribute:: mbgdml.data.dataset.dataSet.E_max

.. autoattribute:: mbgdml.data.dataset.dataSet.F

.. note::
    There is no explicit force unit attribute in a data set.
    It is assumed to be ``e_unit`` ``r_unit``:sup:`-1`.

.. autoattribute:: mbgdml.data.dataset.dataSet.F_mean

.. autoattribute:: mbgdml.data.dataset.dataSet.F_min

.. autoattribute:: mbgdml.data.dataset.dataSet.F_max

.. autoattribute:: mbgdml.data.dataset.dataSet.theory

Loading and saving
------------------

Data sets are stored as NumPy ``.npz`` files.
To load a data set, you can pass the path to a ``dataset.npz`` file or explicitly use the :func:`~mbgdml.data.dataset.dataSet.load` function.

.. code-block:: python

    from mbgdml.data import dataSet

    dset = dataSet('./path/to/dataset.npz')
    # Or
    dset = dataSet()
    dset.load('./path/to/dataset.npz')


.. automethod:: mbgdml.data.dataset.dataSet.load

Saving a data set can be done using the :func:`mbgdml.data.dataset.dataSet.save` function.
The required ``data`` dictionary for ``save`` is provided as the ``dataset`` attribute which creates a dictionary of all data to save in the ``npz`` file.

.. code-block:: python

    dset.save('dataset', dset.dataset, './path/to')

.. automethod:: mbgdml.data.dataset.dataSet.save





Creation
--------

TODO

.. automethod:: mbgdml.data.dataset.dataSet.add_pes_data

Unit conversion
---------------

mbGDML provides a simple way to convert Cartesian coordinates, energies, or forces to a variety of units.

.. automethod:: mbgdml.data.dataset.dataSet.convertR

.. automethod:: mbgdml.data.dataset.dataSet.convertE

.. automethod:: mbgdml.data.dataset.dataSet.convertF

So, say we wanted to convert the energies and forces of ``my_dataset`` to kcal/mol and kcal/mol/A.
The coordinates are already in Angstroms, so we just need to convert the energies and forces.

.. code-block:: python
    
    my_dataset.convertE('kcal/mol')
    my_dataset.convertF('hartree', 'bohr', 'kcal/mol', 'Angstrom')

.. warning::

    ``convertF`` does not change any unit specifications (i.e., ``r_unit`` and ``e_unit``), but **needs** to match both coordinate and energy units.




.. _mb-data-sets:

Many-body data sets
-------------------

TODO
