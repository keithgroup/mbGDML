======
Models
======

Models contain information for mbGDML to make energy and force predictions.
Not much functionality is centered around manipulating models, so only basic information is provided.



Loading and saving
------------------

Models are stored as NumPy ``.npz`` files.
To load a data set, you can pass the path to a ``dataset.npz`` file or explicitly use the :func:`~mbgdml.data.model.mbModel.load` function.

.. code-block:: python

    from mbgdml.data import mbModel

    model = mbModel('./path/to/dataset.npz')
    # Or
    model = mbModel()
    model.load('./path/to/dataset.npz')


.. automethod:: mbgdml.data.model.mbModel.load

Saving a model can be done using the :func:`mbgdml.data.model.mbModel.save` function.
The required ``model`` dictionary for ``save`` is provided as the ``model`` attribute which creates a dictionary of all data to save in the ``npz`` file.

.. code-block:: python

    model.save('model', model.model, './path/to')

.. automethod:: mbgdml.data.model.mbModel.save



Contents
--------

Structure set objects contain the following information.

Atomic numbers
^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.model.mbModel.z

.. autoattribute:: mbgdml.data.model.mbModel.n_z

MD5 hash
^^^^^^^^

.. autoattribute:: mbgdml.data.model.mbModel.md5


Version control
^^^^^^^^^^^^^^^

.. autoattribute:: mbgdml.data.model.mbModel.code_version