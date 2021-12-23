======
mbGDML
======

.. image:: https://app.travis-ci.com/keithgroup/mbGDML.svg?branch=main
   :target: https://app.travis-ci.com/keithgroup/mbGDML
   :alt: Travis-CI build status

.. image:: https://codecov.io/gh/keithgroup/mbGDML/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/keithgroup/mbGDML
   :alt: Codecov test coverage

.. image:: https://img.shields.io/lgtm/grade/python/g/keithgroup/mbGDML.svg?logo=lgtm&logoWidth=18
   :target: https://lgtm.com/projects/g/keithgroup/mbGDML/context:python

.. image:: https://img.shields.io/github/license/keithgroup/mbGDML
   :target: https://github.com/keithgroup/mbGDML/blob/master/LICENSE

Many-body gradient-domain machine learning (mbGDML) is a Python package that creates, uses, and analyzes a kernel machine learning potential based on the many-body expansion.

**Disclaimer:** This package is still under active development and is not ready for production.

Why?
====

Often a trade-off game is played between accuracy confidence and computational cost with molecular simulations.
Machine learning (ML) force fields attempt to offer a compromise somewhere between quantum chemistry and classical force fields.
However, the trade-off game we play with ML force fields is between transferability (i.e., applicability of a trained model to a different system) and the amount of training data.

.. image:: images/ml-force-field.svg
   :width: 400px
   :align: center

Using high-levels of quantum chemistry (e.g., coupled cluster or configuration interaction) dramatically limits the amount of calculations one can reasonably perform.
`Gradient domain machine learning (GDML) <http://quantum-machine.org/gdml/>`_ is one example of a ML force field designed to be data efficient---only requiring hundreds of training data points.
GDML accomplishes this by learning the fundamental relationship between a geometry and its atomic forces (i.e., gradient).

.. figure:: images/gdml-concept.png
   :width: 350px
   :align: center

   Chmiela, S.; et al. *Sci. Adv.* **2017**, 3 (5), e1603015. DOI: `10.1126/sciadv.1603015 <https://doi.org/10.1126/sciadv.1603015>`_

However, GDML is not inherently size or species transferable.
Meaning changing the number or types of atoms is not allowed after a GDML model is trained---making simulations on arbitrarily sized systems like solvents futile.
To circumvent this limitation, we developed a many-body approach where GDML learns *n*-body interactions to have an efficiently trained, transferable ML potential.

.. image:: images/explicit-water-methanol-mbe-allorders.svg
   :width: 350px
   :align: center

.. toctree::
   :hidden:

   Install <install>
   Data and Models <data-and-models/data-formats>
   QC Calculations <qc-calcs>
   Training <training>
   Tutorials <tutorials/tutorials>
   API <doc/modules>
   contributing
