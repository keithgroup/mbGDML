.. _structure-sets:

==============
Structure sets
==============

A structure set (commonly abbreviated as ``rset`` or ``Rset``) represents the starting point of mbGDML.
They are used to store structures from the same process or procedure; for example, a molecular dynamics trajectory or a single generated conformers.
No data regarding energies or forces are stored in the structure set.
A single or multiple ``rset`` may then be used to curate :ref:`data sets<data-sets>` for mbGDML training.


.. _load-save-rset:

Loading and saving
------------------

To load a structure set, you can pass the path to a ``structureset.npz`` file or explicitly use the :func:`~mbgdml.data.structureset.structureSet.load` function.

.. code-block:: python

    from mbgdml.data import structureSet

    rset = structureSet('./path/to/structureset.npz')
    # Or
    rset = structureSet()
    rset.load('./path/to/structureset.npz')


.. automethod:: mbgdml.data.structureset.structureSet.load
    :noindex:

Saving a data set is just as easy.

.. code-block:: python

    rset.save('structureset', rset.structureset, './path/to')

.. automethod:: mbgdml.data.structureset.structureSet.save
    :noindex:



Creation
--------

Structure sets require :ref:`atomic numbers<Atomic numbers>`, :ref:`Cartesian coordinates` (and :attr:`~mbgdml.data.basedata.mbGDMLData.r_unit`), :ref:`Entity IDs`, and :ref:`Component IDs` to create.
The easiest way is to assign the data to the corresponding attributes.

Files commonly used in computational chemistry already have all these data.
We provide a few ways to directly parse these data directly from the file instead of manually adding them to the structure set.

.. automethod:: mbgdml.data.structureset.structureSet.from_xyz
    :noindex:

.. automethod:: mbgdml.data.structureset.structureSet.from_npz
    :noindex:

Creating entity and component IDs for single-component structures we provide two structures to help initialize these data.

.. autofunction:: mbgdml.utils.get_entity_ids
    :noindex:

.. autofunction:: mbgdml.utils.get_comp_ids
    :noindex:



Contents
--------

Structure set objects contain the following information.

Atomic numbers
^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.structureset.structureSet.z
    :noindex:

.. autoattribute:: mbgdml.data.structureset.structureSet.n_z
    :noindex:

Cartesian coordinates
^^^^^^^^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.structureset.structureSet.R
    :noindex:

.. autoattribute:: mbgdml.data.structureset.structureSet.n_R
    :noindex:

.. autoattribute:: mbgdml.data.structureset.structureSet.r_unit
    :noindex:

MD5 hash
^^^^^^^^

.. autoattribute:: mbgdml.data.structureset.structureSet.md5
    :noindex:

.. note::
   MD5 hashes are recomputed whenever the attribute is called.

Entity IDs
^^^^^^^^^^

.. autoattribute:: mbgdml.data.structureset.structureSet.entity_ids
    :noindex:

Component IDs
^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.structureset.structureSet.comp_ids
    :noindex:
