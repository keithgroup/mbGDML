Installation
============

At the moment, the only way to install mbGDML is directly from the `GitHub repository <https://github.com/keithgroup/mbGDML>`_.

Requirements
############

The following packages are required:

* ase
* cclib (>=1.6.4)
* click
* dscribe
* mako
* matplotlib
* natsort
* numpy
* periodictable
* sgdml

All of these required packages can be installed with:

::

    pip install ase 'cclib>=1.6.4' click dscribe mako matplotlib natsort numpy periodictable sgdml


Then, clone and install the repository.

::

    git clone https://github.com/keithgroup/mbGDML
    cd mbgdml
    pip install .
