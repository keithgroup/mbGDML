======
Models
======

Models contain information for mbGDML to make energy and force predictions.
Not much functionality is centered around manipulating models, so only basic information is provided.


.. _load-save-model:

Loading and saving
------------------

Models are stored as NumPy ``.npz`` files.
To load a data set, you can pass the path to a ``model.npz`` file or explicitly use the :meth:`~mbgdml.models.gdmlModel.load` function.

.. code-block:: python

    from mbgdml.models import gdmlModel

    model = gdmlModel('./path/to/model.npz')
    # Or
    model = gdmlModel()
    model.load('./path/to/model.npz')

Saving a model can be done using the :meth:`~mbgdml.models.gdmlModel.save` function.
The required ``model`` dictionary for ``save`` is provided as the ``model`` attribute which creates a dictionary of all data to save in the ``npz`` file.

.. code-block:: python

    model.save('./path/to/model.npz')



Contents
--------

Structure set objects contain the following information.

Atomic numbers
^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.models.gdmlModel.Z
    :noindex:

.. autoattribute:: mbgdml.models.gdmlModel.n_Z
    :noindex:

MD5 hash
^^^^^^^^

.. autoattribute:: mbgdml.models.gdmlModel.md5
    :noindex:


Version control
^^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.models.gdmlModel.code_version
    :noindex:
