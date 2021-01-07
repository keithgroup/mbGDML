Data sets
=========

Loading and saving
------------------

Data sets are stored as NumPy ``.npz`` files. To load a data set, you can pass
the path to the ``dataset.npz`` file or explicitly use the
:func:`~mbgdml.data.dataset.mbGDMLDataset.load` function.

.. code-block:: python

    from mbgdml.data import mbGDMLDataset

    my_dataset = mbGDMLDataset(path='./path/to/dataset.npz')
    # Or
    my_dataset = mbGDMLDataset()
    my_dataset.load('./path/to/dataset.npz')


.. automethod:: mbgdml.data.dataset.mbGDMLDataset.load

Saving a data set is just as easy.

.. code-block:: python

    my_dataset.save('dataset', my_dataset.dataset, './path/to')

.. automethod:: mbgdml.data.dataset.mbGDMLDataset.save


Input formats
-------------

.. _xyz-data-sets:

XYZ files
^^^^^^^^^

The most straightforward way to create a data set is from a set of xyz data
files. sGDML models require, at the bare minimum, Cartesian coordinates and 
atomic forces. Including energies is not required, but highly recommended.

Suppose there is an xyz file ``4h2o.traj`` that contains the molecular dynamics
(MD) trajectory of a water tetramer (4mer) in Angstroms and electronic energies
of each structure as a comment in Hartrees (Eh).

.. code-block::

    12
    -305.29934495
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
    -305.29555214
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
    ...

This would provide both Cartesian coordinates and energies for our data set.
We can this use the :func:`~mbgdml.data.dataset.mbGDMLDataset.read_xyz`
function to 

.. code-block:: python

    from mbgdml.data import mbGDMLDataset

    my_dataset = mbGDMLDataset()
    my_dataset.read_xyz(
        './4h2o.traj', 'coords', r_unit='Angstrom', e_unit='hartree',
        energy_comments=True
    )

.. automethod:: mbgdml.data.dataset.mbGDMLDataset.read_xyz


Now, we just need to include the forces from the file ``4h2o.forces`` (Eh/Bohr):

.. code-block::

    12
    
    O     -0.0783514608      0.0088741724     -0.0193159932
    H      0.0351648371      0.0006203498     -0.0202709745
    H      0.0405428086     -0.0097068830      0.0380108300
    O      0.0167995790     -0.0245643408      0.0080700297
    H     -0.0046242192      0.0162589761      0.0047074502
    H     -0.0126375089      0.0083850560     -0.0138768343
    O     -0.0068327390      0.0398848084     -0.0054261845
    H     -0.0109068224     -0.0224998064      0.0076864007
    H      0.0191726508     -0.0168322451     -0.0001864401
    O     -0.0075486349     -0.0128541591     -0.0008749207
    H     -0.0015802679      0.0001992652      0.0063942910
    H      0.0108017777      0.0122348067     -0.0049176542
    12
    
    O     -0.1157388144      0.0139375032     -0.0291447198
    H      0.0504769896     -0.0000680118     -0.0296009259
    H      0.0601598658     -0.0140470382      0.0558754800
    O      0.0285954317     -0.0415674024      0.0133251427
    H     -0.0091421073      0.0252994099      0.0053522204
    H     -0.0205434864      0.0165061826     -0.0207193992
    O     -0.0096676194      0.0602133630     -0.0071054884
    H     -0.0160102577     -0.0346994508      0.0117712150
    H      0.0284465643     -0.0244439114     -0.0009052223
    O     -0.0101593743     -0.0173954659     -0.0013757060
    H     -0.0025551540     -0.0000676045      0.0095601028
    H      0.0161379621      0.0163324264     -0.0070326993
    ...

which can be done like so.

.. code-block:: python

    my_dataset.read_xyz('./4h2o.forces', 'forces')

For ease of use, these two files can be combined into an extended xyz format
where the forces are listed after the Cartesian coordinates like so.

.. code-block::

    12
    -305.29934495
    O      1.5302902756      1.1157449133      0.3982563570     -0.0783514608      0.0088741724     -0.0193159932
    H      2.3549901725      1.1308050474     -0.0579131923      0.0351648371      0.0006203498     -0.0202709745
    H      0.8132702485      1.2825167543     -0.3164003435      0.0405428086     -0.0097068830      0.0380108300
    O     -1.4624673812     -1.1623606980     -0.4446728115      0.0167995790     -0.0245643408      0.0080700297
    H     -1.3991198038     -2.0179253584     -0.9024790666     -0.0046242192      0.0162589761      0.0047074502
    H     -0.8480294858     -1.2751883405      0.3337952128     -0.0126375089      0.0083850560     -0.0138768343
    O     -0.4879631261      1.1762906002     -1.4407893150     -0.0068327390      0.0398848084     -0.0054261845
    H     -0.9105866803      0.3308684519     -1.2273871952     -0.0109068224     -0.0224998064      0.0076864007
    H     -1.2082963784      1.8347953332     -1.4343826167      0.0191726508     -0.0168322451     -0.0001864401
    O      0.4412097546     -1.1258961267      1.4972804137     -0.0075486349     -0.0128541591     -0.0008749207
    H      0.2625510495     -0.9515213602      2.4228957998     -0.0015802679      0.0001992652      0.0063942910
    H      0.9141493552     -0.3381292166      1.1717977575      0.0108017777      0.0122348067     -0.0049176542
    12
    -305.29555214
    O      1.5380481387      1.1185759295      0.4031603166     -0.1157388144      0.0139375032     -0.0291447198
    H      2.3548283904      1.1215564005     -0.0514567537      0.0504769896     -0.0000680118     -0.0296009259
    H      0.8007302238      1.2863389508     -0.3291414956      0.0601598658     -0.0140470382      0.0558754800
    O     -1.4580502952     -1.1613686393     -0.4389733627      0.0285954317     -0.0415674024      0.0133251427
    H     -1.3726832972     -2.0234475128     -0.8945355690     -0.0091421073      0.0252994099      0.0053522204
    H     -0.8369199082     -1.2963126015      0.3429271961     -0.0205434864      0.0165061826     -0.0207193992
    O     -0.4838151640      1.1738421118     -1.4413091990     -0.0096676194      0.0602133630     -0.0071054884
    H     -0.9024855217      0.3307130603     -1.2447990479     -0.0160102577     -0.0346994508      0.0117712150
    H     -1.2119858522      1.8340985256     -1.4187444976      0.0284465643     -0.0244439114     -0.0009052223
    O      0.4448622097     -1.1232759530      1.4970578637     -0.0101593743     -0.0173954659     -0.0013757060
    H      0.2627545300     -0.9398674788      2.4187400482     -0.0025551540     -0.0000676045      0.0095601028
    H      0.8833158282     -0.3236308577      1.1613739862      0.0161379621      0.0163324264     -0.0070326993
    ...

.. note::
    Extended xyz formats are assumed to have forces.

Then you can load all the data at once.

.. code-block:: python

    my_dataset.read_xyz('./4h2o.extxyz', 'extended')

.. _output-data-sets:

Output files
^^^^^^^^^^^^

A common routine is to partition structures from larger ones (e.g., dimers from
a single tetramer). This means that the energy and forces of all the new 
partitions need to be recalculated. Since many computational chemistry packages
allow multiple calculations in a single job, mbGDML provides a simple way to
create a data set directly from the output file.

.. tip::
    To parse data from computational chemistry output files see
    :doc:`partitioning`.

Here is an example with the following `ORCA 4.2.0 output file
<https://raw.githubusercontent.com/keithgroup/mbGDML/master/tests/data/
partition-calcs/out-4H2O-300K-1-ABC.out>`_ for a water trimer from a tetramer.

.. code-block:: python

    partition_calc = data.PartitionOutput(
        './path/to/out-4H2O-300K-1-ABC.out',
        '4H2O',
        'ABC',
        300,
        'hartree',
        'bohr',
        md_iter=1,
        theory='mp2.def2tzvp'
    )
    
    test_dataset = data.mbGDMLDataset()
    test_dataset.from_partitioncalc(partition_calc)

.. automethod:: mbgdml.data.dataset.mbGDMLDataset.from_partitioncalc

Unit conversion
---------------

mbGDML provides a simple way to convert Cartesian coordinates, energies, or 
forces to a variety of units.

.. automethod:: mbgdml.data.dataset.mbGDMLDataset.convertR

.. automethod:: mbgdml.data.dataset.mbGDMLDataset.convertE

.. automethod:: mbgdml.data.dataset.mbGDMLDataset.convertF

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

.. automethod:: mbgdml.data.dataset.mbGDMLDataset.from_combined


Many-body data sets
-------------------

Training n-body GDML model requires a data set with all lower-order
contributions removed. For example, to prepare a 2-body data set we have to 
remove all 1-body contributions from our dimer (2mer) data set. This is
accomplished by first :doc:`training a sGDML model<training>` on monomers
(1mers) then preparing the 2-body data set like so.

.. code-block:: python

    from mbgdml.data import mbGDMLDataset

    # Load the dimer data set.
    my_2mer_dataset = data.mbGDMLDataset(path='./path/to/2mer-dataset.npz')

    # Create the 2-body data set.
    my_mb_dataset = data.mbGDMLDataset()
    my_mb_dataset.create_mb(my_2mer_dataset, ['./path/to/1mer-model.npz'])

.. automethod:: mbgdml.data.dataset.mbGDMLDataset.create_mb

.. warning::

    Each mbGDML model is dependent on the ones used to prepare the many-body 
    data set. For example, a 3-body GDML model can only be used with the 1-body
    and 2-body models used to create the many-body data set.

Available data
----

The following data are available from data sets.

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.z

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.R

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.r_unit

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.E

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.e_unit

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.E_max

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.E_mean

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.E_min

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.E_var

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.F

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.F_max

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.F_mean

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.F_min

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.F_var

.. autoattribute:: mbgdml.data.dataset.mbGDMLDataset.md5