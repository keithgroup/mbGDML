#!/usr/bin/env python
# MIT License
# 
# Copyright (c) 2020, Alex M. Maldonado
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tests for `mbgdml` package."""

import os
import pytest
import numpy as np

import mbgdml.data as data
import mbgdml.utils as utils

# Must be run from mbGDML root directory.

def test_data_create_dataset():
    ref_output_path = 'tests/data/partition-calcs/out-4H2O-300K-1-ABC.out'
    write_path = 'tests/data/write/'

    test_partition = data.PartitionCalcOutput(ref_output_path)
    dataset = data.mbGDMLDataset()
    dataset.partition_dataset_name(
        test_partition.partition,
        test_partition.cluster,
        test_partition.temp,
        test_partition.iter,
    )
    dataset.create_dataset(
        write_path,
        dataset.dataset_name,
        test_partition.atoms,
        test_partition.coords,
        test_partition.energies,
        test_partition.forces,
        'kcal/mol',
        'hartree',
        'bohr',
        theory='MP2.def2-TZVP',
        write=False
    )

    assert dataset.base_vars['r_unit'] == 'Angstrom'
    assert np.allclose(dataset.base_vars['R'][0][4],
                       np.array([-1.39912, -2.017925, -0.902479]))

    assert dataset.base_vars['e_unit'] == 'kcal/mol'
    assert dataset.base_vars['E'][0] == -143672.88767989643
    assert dataset.base_vars['E_max'] == -143668.55663397504
    assert dataset.base_vars['E_mean'] == -143671.36616500877
    assert dataset.base_vars['E_min'] == -143674.07605733958
    assert dataset.base_vars['E_var'] == 1.1866207743053436

    assert dataset.base_vars['F_max'] == 75.06988364013725
    assert np.allclose(dataset.base_vars['F'][30][0],
                       np.array([6.81548275, -14.34210238, -63.49876045]))
    assert dataset.base_vars['F_mean'] == -2.1959648981851178e-08
    assert dataset.base_vars['F_min'] == -75.04996184655204
    assert dataset.base_vars['F_var'] == 373.33604029702724

    assert dataset.base_vars['e_unit'] == 'kcal/mol'
    assert dataset.base_vars['name'] == 'ABC-4H2O-300K-1-dataset'
    assert dataset.base_vars['system'] == 'solvent'
    assert dataset.base_vars['solvent'] == 'water'
    assert dataset.base_vars['cluster_size'] == 3
    assert dataset.base_vars['theory'] == 'MP2.def2-TZVP'


def test_data_create_predictset():

    dataset_path = 'tests/data/datasets/4H2O-2mer-dataset.npz'
    model_paths = [
        'tests/data/models/4H2O-1mer-model-MP2.def2-TZVP-train300-sym2.npz',
        'tests/data/models/4H2O-2body-model-MP2.def2-TZVP-train300-sym8.npz'
    ]

    test_predictset = data.mbGDMLPredictset()
    test_predictset.load_models(model_paths)
    test_predictset.load_dataset(dataset_path)

    # Reducing number of data to the first five structures
    test_predictset.dataset['R'] = test_predictset.dataset['R'][0:5, :, :]
    test_predictset.dataset['E'] = test_predictset.dataset['E'][0:5]
    test_predictset.dataset['F'] = test_predictset.dataset['F'][0:5, :, :]

    test_predictset.create_predictset()

    #TODO Add assert statements

