====================
Many-body expansions
====================

The many-body expansion (MBE) represents the total system energy, :math:`E`, composed of :math:`N` noncovalently connected (i.e., non-intersecting) fragments/monomers as the sum of :math:`n`-body interaction energies:

.. math::
    E = \sum_{i}^N E_i^{(1)} + \sum_{i < j}^N \Delta E_{ij}^{(2)} + \sum_{i < j < k}^N \Delta E_{ijk}^{(3)} + \cdots.
    :label: mbe_sum

Here, :math:`N` is the number of monomers; :math:`i`, :math:`j`, and :math:`k` are monomer indices; :math:`E_i^{(1)}` is the energy of monomer :math:`i`; and :math:`\Delta E_{\; i, \: j, \: \ldots}^{\; (n)}` represents the :math:`n`-body interaction energy contribution of the structure containing fragments :math:`i`, :math:`j`, :math:`\ldots` with lower order (:math:`< n`) contributions removed.
For example, the 2-body contribution of :math:`i` and :math:`j` is

.. math::
    \Delta E_{ij}^{(2)} = E_{ij}^{(2)} - E_{i}^{(1)} - E_{j}^{(1)},
    :label: 2_body_term

and the 3-body contributions with monomers :math:`i`, :math:`j`, and :math:`k` are

.. math::
    \Delta E_{ijk}^{(3)} = E_{ijk}^{(3)} - \Delta E_{ij}^{(2)} - \Delta E_{ik}^{(2)} - \Delta E_{jk}^{(2)}  - E_{i}^{(1)} - E_{j}^{(1)} - E_{k}^{(1)}.
    :label: 3_body_term

Equation :eq:`mbe_sum` is exact when all :math:`n`-body contributions up to :math:`N` are accounted for with exact accuracy and precision.
This equation also holds for properties that can be expressed as a derivative of energy (e.g., gradients).
In practice, the expansion is always truncated to some order, :math:`n`, significantly less than :math:`N`.

What is a fragment?
===================

MBEs are built upon what we define our fragments/monomers to be.
The most straightforward procedure is to define each molecule as its own fragment.
However, nothing precludes intersecting fragments where a single molecule is broken up into multiple fragments.
This comes with the additional complexity of capping groups which will not be discussed here.
Atomic pairwise potentials where fragments are individual atoms is also possible.

The choice is yours, but the driving aspect of selecting a fragmentation scheme is often the number and size of :math:`n`-body combinations.
If we consider a water trimer, there are substantially more 2- and 3-body structures if we fragment with respect to atoms instead of molecules.
Not to mention that whole molecular fragments are typically more accurate.
Computational cost of fragments also becomes an issue if they are particularly large.

Water trimer example
====================

We find MBEs are much easier to understand with an example.
Suppose we want to compute the energy of the following trimer with MP2/def2-TZVP and that this level of theory is intractable.

.. image:: images/mbe-explained/3h2o-energy.png
    :width: 200px
    :align: center

Obviously this calculation is possible in a matter of seconds, but just imagine we cannot.

1-body prediction
-----------------

The first term of the MBE (Equation :eq:`mbe_sum`) represents the 1-body contributions of this structure.
Before this we have to define what our 1-body structures or fragments are.
Each molecule will be a fragment since they are small and it is a noncovalent cluster (fragment indices are shown above).
Thus, our 1-body prediction, :math:`E^{(1)}`, of this trimer would be

.. math::
    E^{(1)} = E_0^{(1)} + E_1^{(1)} + E_2^{(1)}.

.. image:: images/mbe-explained/3h2o-energy-1bodies.png
    :width: 200px
    :align: center

Our sum becomes

.. math::
    E^{(1)} = -76.31270 \;\text{Eh} + -76.31251 \;\text{Eh} + -76.31273 \;\text{Eh} = -228.93794 \;\text{Eh}.

Our 1-body prediction has an error of -0.02504 Eh (-15.7 kcal/mol).
This error is unsurprising as we have made not accounted for how these water molecules are interacting.

2-body prediction
-----------------

By including monomer interactions (e.g., 2-body) we can substantially reduce the error and cost of ML potentials.

.. image:: images/mbe-explained/3h2o-energy-2body-0,1.png
    :width: 450px
    :align: center

.. image:: images/mbe-explained/3h2o-energy-2body-0,2.png
    :width: 450px
    :align: center

.. image:: images/mbe-explained/3h2o-energy-2body-1,2.png
    :width: 450px
    :align: center

Our 2-body contribution/correction is

.. math::
    \Delta E^{(2)} = \Delta E_{0,1}^{(2)} + \Delta E_{0,2}^{(2)} + \Delta E_{1,2}^{(2)}.

For our particular system, this ends up being 

.. math::
    \Delta E^{(2)} = -0.00831  \;\text{Eh} + -00705  \;\text{Eh} + -0.00700  \;\text{Eh} = -0.02236 \;\text{Eh}.

Thus, our 2-body prediction is the original 1-body contribution plus the 2-body correction,

.. math::
    E^{(2)} = -228.93794 \;\text{Eh} + -0.02236 \;\text{Eh} = -228.96033 \;\text{Eh}.

Our 2-body prediction has an error of -0.00267 Eh (-1.7 kcal/mol).

Additional resources
--------------------

Please see the following resources for additional information on many-body expansions:



Specifying fragments in mbGDML
==============================

In order to make many-body predictions we have to specify the fragments to generate :math:`n`-body combinations from.
For example, we can consider making a MBE(2) prediction of a water and methanol cluster.
This is rather small system, but it is useful for understanding the concepts.

.. figure:: images/mbe-explained/2-body-example.png
   :width: 250px
   :align: center

   Example structure of a water and methanol molecule.

As with most atomistic modeling practices, we must specify atomic numbers and coordinates of the structure.

.. figure:: images/mbe-explained/2-body-z.png
   :width: 250px
   :align: center

   ``Z``: atomic numbers of all atoms in the system.

.. figure:: images/mbe-explained/2-body-r.png
   :width: 525px
   :align: center

   ``R``: Cartesian coordinates in the same order as ``Z``.

Some ML potentials require the order of the atoms to be the exact same (e.g., GDML).
This means any indistinguishable atoms must be in the same order.
For the structure specified above, we must have the water molecule then the methanol.
Water's oxygen atom must come before the hydrogens whose order does not matter.
With methanol, we specify the OH group first, then the CH3 group where the first hydrogen is the one furthest from the OH hydrogen and proceeding in a clockwise direction.

.. figure:: images/mbe-explained/2-body-entity-ids.png
   :width: 250px
   :align: center

   ``entity_ids``: integers that specify which fragment each atom belongs to.

.. figure:: images/mbe-explained/2-body-comp-ids.png
   :width: 175px
   :align: center

   ``comp_ids``: labels for each ``entity_id`` used to determine relevant models.



