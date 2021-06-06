Structure sets
=========

A structure set (commonly abbreviated as ``rset`` represents the starting point
of mbGDML. The purpose of these NumPy ``npz`` files is to store structures.
Particularly, be a source of structures to sample for data sets or even for
analyses. A structure set can contain a single or multiple structures.

Loading and saving
------------------

Structure sets are stored as NumPy ``.npz`` files. To load a structure set, you
can pass the path to a ``structureset.npz`` file or explicitly use the
:func:`~mbgdml.data.structureset.structureSet.load` function.

.. code-block:: python

    from mbgdml.data import structureSet

    rset = structureSet('./path/to/structureset.npz')
    # Or
    rset = structureSet()
    rset.load('./path/to/structureset.npz')


.. automethod:: mbgdml.data.structureset.structureSet.load

Saving a data set is just as easy.

.. code-block:: python

    rset.save('structureset', rset.structureset, './path/to')

.. automethod:: mbgdml.data.structureset.structureSet.save


Creating
--------

Currently the only supported structure set creation method is using 
:func:`~mbgdml.data.structureset.structureSet.from_xyz`.

.. automethod:: mbgdml.data.structureset.structureSet.from_xyz

Each structure set at the bare minium requires three pieces of information:
:ref:`Coordinates`, :ref:`Entity IDs`, and :ref:`Component IDs`.
Each will be explained in the following sections.

Coordinates
^^^^^^^^^^^

The most straightforward way to input atomic coordinates and create a structure
set is from XYZ files. These XYZ files can contain a single or multiple
individual XYZ structures. Every XYZ structure will be added to the structure
set. For example, below is an XYZ file for two water tetramers.

.. code-block::

    12
    
    O      1.5302902756      1.1157449133      0.3982563570
    H      2.3549901725      1.1308050474     -0.0579131923
    H      0.8132702485      1.2825167543     -0.3164003435
    O     -1.4624673812     -1.1623606980     -0.4446728115
    H     -1.3991198038     -2.0179253584     -0.9024790666
    H     -0.8480294858     -1.2751883405      0.3337952128
    O     -0.4879631261      1.1762906002     -1.4407893150
    H     -0.9105866803      0.3308684519     -1.2273871952
    H     -1.2082963784      1.8347953332     -1.4343826167
    O      0.4412097546     -1.1258961267      1.4972804137
    H      0.2625510495     -0.9515213602      2.4228957998
    H      0.9141493552     -0.3381292166      1.1717977575
    12
    
    O      1.5380481387      1.1185759295      0.4031603166
    H      2.3548283904      1.1215564005     -0.0514567537
    H      0.8007302238      1.2863389508     -0.3291414956
    O     -1.4580502952     -1.1613686393     -0.4389733627
    H     -1.3726832972     -2.0234475128     -0.8945355690
    H     -0.8369199082     -1.2963126015      0.3429271961
    O     -0.4838151640      1.1738421118     -1.4413091990
    H     -0.9024855217      0.3307130603     -1.2447990479
    H     -1.2119858522      1.8340985256     -1.4187444976
    O      0.4448622097     -1.1232759530      1.4970578637
    H      0.2627545300     -0.9398674788      2.4187400482
    H      0.8833158282     -0.3236308577      1.1613739862


This would provide the Cartesian coordinates (in Angstroms) for our structure
set.

Entity IDs
^^^^^^^^^^

TODO

Component IDs
^^^^^^^^^^^^^

TODO

Unit conversion
---------------

mbGDML provides a simple way to convert Cartesian coordinates, energies, or 
forces to a variety of units.

.. automethod:: mbgdml.data.dataset.dataSet.convertR

.. automethod:: mbgdml.data.dataset.dataSet.convertE

.. automethod:: mbgdml.data.dataset.dataSet.convertF

So, say we wanted to convert the energies and forces of ``my_dataset`` to 
kcal/mol and kcal/mol/A. The coordinates are already in Angstroms, so we just
need to convert the energies and forces.

.. code-block:: python
    
    my_dataset.convertE('kcal/mol')
    my_dataset.convertF('hartree', 'bohr', 'kcal/mol', 'Angstrom')

.. warning::

    ``convertF`` does not change any unit specifications (i.e., ``r_unit`` and 
    ``e_unit``), but **needs** to match both coordinate and energy units.


Combining data sets
-------------------

There are many times where you would want to combine one data set with another;
for example, multiple MD simulations or partitions.

.. note::

    The data sets can only be combined if they are the same system and units.
    Meaning the same number and order of atoms, units, and array dimensions.

.. automethod:: mbgdml.data.dataset.dataSet.from_combined


Many-body data sets
-------------------

Training n-body GDML model requires a data set with all lower-order
contributions removed. For example, to prepare a 2-body data set we have to 
remove all 1-body contributions from our dimer (2mer) data set. This is
accomplished by first :doc:`training a sGDML model<training>` on monomers
(1mers) then preparing the 2-body data set like so.

.. code-block:: python

    from mbgdml.data import dataSet

    # Load the dimer data set.
    my_2mer_dataset = data.dataSet(path='./path/to/2mer-dataset.npz')

    # Create the 2-body data set.
    my_mb_dataset = data.dataSet()
    my_mb_dataset.create_mb(my_2mer_dataset, ['./path/to/1mer-model.npz'])

.. automethod:: mbgdml.data.dataset.dataSet.create_mb

.. warning::

    Each mbGDML model is dependent on the ones used to prepare the many-body 
    data set. For example, a 3-body GDML model can only be used with the 1-body
    and 2-body models used to create the many-body data set.

Available data
----

The following data are available from data sets.

.. autoattribute:: mbgdml.data.dataset.dataSet.z

.. autoattribute:: mbgdml.data.dataset.dataSet.R

.. autoattribute:: mbgdml.data.dataset.dataSet.r_unit

.. autoattribute:: mbgdml.data.dataset.dataSet.E

.. autoattribute:: mbgdml.data.dataset.dataSet.e_unit

.. autoattribute:: mbgdml.data.dataset.dataSet.E_max

.. autoattribute:: mbgdml.data.dataset.dataSet.E_mean

.. autoattribute:: mbgdml.data.dataset.dataSet.E_min

.. autoattribute:: mbgdml.data.dataset.dataSet.E_var

.. autoattribute:: mbgdml.data.dataset.dataSet.F

.. autoattribute:: mbgdml.data.dataset.dataSet.F_max

.. autoattribute:: mbgdml.data.dataset.dataSet.F_mean

.. autoattribute:: mbgdml.data.dataset.dataSet.F_min

.. autoattribute:: mbgdml.data.dataset.dataSet.F_var

.. autoattribute:: mbgdml.data.dataset.dataSet.md5