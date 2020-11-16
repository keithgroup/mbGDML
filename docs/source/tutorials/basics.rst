Basic concepts
==============

This tutorial covers, well, the basics of many-body gradient domain machine
learning (mbGDML) force fields. Please read through this before moving to other
tutorials.

Typical workflow
----------------

This section will walk through the various steps involved in creating a 3-body 
GDML force field for water.

.. note::

    A 3-body GDML force field is the combination of 1-body, 2-body, and 3-body
    GDML models.

Geometric sampling
^^^^^^^^^^^^^^^^^^

Machine learning methods are data driven and rely on diverse sampling of the
potential energy surface (PES) regions of interest. Ultimately, a 3-body GDML
force field requires **structures**, **energies**, and **atomic forces** for
water monomers (1mers), dimers (2mers), and trimers (3mers).
Molecular dynamics (MD) is one convenient and quick technique to sample
physically-relevant equilibrium and nonequilibrium configurations.

.. Add information about different temperatures.

How you generate these data is up to you and what is best for your system.
For this example, one approach is to partition the various sized water clusters
from a set of tetramer (4mer) MD simulations. That is, running several 4mer MD
simulations, carving all possible 1mer, 2mer, and 3mer structures, and
recalculating energies and forces. This procedure would provide a total of four
1mer, six 2mer, and four 3mer structures per MD time step.


.. tip::

    Any package can be used to run MD simulations or calculate energies and
    forces. The data just needs to be in :ref:`output files parsable by cclib
    <output-data-sets>` or in :ref:`xyz format<xyz-data-sets>`.

Preparing data sets and training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have 

TODO

Using mbGDML force fields
^^^^^^^^^^^^^^^^^^^^^^^^^

TODO