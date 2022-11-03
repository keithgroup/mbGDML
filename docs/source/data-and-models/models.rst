======
Models
======

Models contain information for mbGDML to make energy and force predictions.
Not much functionality is centered around manipulating models, so only basic information is provided.


.. _load-save-model:

Loading and saving
------------------

Models are stored as NumPy ``.npz`` files.
To load a data set, you can pass the path to a ``model.npz`` file or explicitly use the :meth:`~mbgdml.data.mbModel.load` function.

.. code-block:: python

    from mbgdml.data import mbModel

    model = mbModel('./path/to/model.npz')
    # Or
    model = mbModel()
    model.load('./path/to/model.npz')

Saving a model can be done using the :meth:`~mbgdml.data.mbModel.save` function.
The required ``model`` dictionary for ``save`` is provided as the ``model`` attribute which creates a dictionary of all data to save in the ``npz`` file.

.. code-block:: python

    model.save('model', model.model, './path/to')



Contents
--------

Structure set objects contain the following information.

Atomic numbers
^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.model.mbModel.Z
    :noindex:

.. autoattribute:: mbgdml.data.model.mbModel.n_Z
    :noindex:

MD5 hash
^^^^^^^^

.. autoattribute:: mbgdml.data.model.mbModel.md5
    :noindex:


Version control
^^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.model.mbModel.code_version
    :noindex:
