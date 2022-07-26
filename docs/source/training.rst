========
Training
========

The goal of training ML models is to optimize regression coefficients (i.e., parameters) to reduce some performance metric.
Once we have a :ref:`data set<data-sets>` with structures, forces, and possibly energies, we can begin training a GDML model.
This is done through the :class:`mbgdml.train.mbGDMLTrain` class.

After initializing a ``mbGDMLTrain`` object, you then load the desired data set using the following function.
This class provides several training routines.

Hyperparameter optimization
===========================

Identifying the optimal hyperparameters for ML models is crucial.
For GDML models, the hyperparameter of interest is the kernel length scale or ``sigma``.
There are several possible procedures for tuning hyperparameters and mbGDML provides two options.

- :meth:`~mbgdml.train.mbGDMLTrain.bayes_opt`
- :meth:`~mbgdml.train.mbGDMLTrain.grid_search`


Iterative training
==================

Curating the training set is no easy task.
Instead of sampling based on energy distributions, we can iteratively build the training set to improve global accuracy using :meth:`~mbgdml.train.mbGDMLTrain.iterative_train`.
This involves two stages:

1. Training a preliminary model with the standard sampling procedure.
For example, a model trained on 200 data points.

2. Using the previous model, make predictions of the entire dataset and identify structures with high errors.
Add these structures to the previous training set and repeat until the final training set size is reached.

In the end, we have a model that has minimized the maximum errors across the data set.
However, this comes at the expense of raising the smallest errors.
