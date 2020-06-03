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
from math import isclose
import pytest
import numpy as np

import mbgdml.data as data
import mbgdml.utils as utils

# Must be run from mbGDML root directory.

def test_data_create_dataset():
    ref_output_path = 'tests/data/partition-calcs/out-4H2O-300K-1-ABC.out'
    write_path = 'tests/data/write/'

    test_partition = data.PartitionOutput(
        ref_output_path,
        '4H2O',
        'ABC',
        300,
        md_iter=1
    )

    test_dataset = data.mbGDMLDataset()

    dataset_name = test_dataset.partition_dataset_name(
        test_partition.cluster_label,
        test_partition.partition_label,
        test_partition.md_temp,
        test_partition.md_iter
    )

    test_dataset.create_dataset(
        write_path,
        dataset_name,
        test_partition.atoms,
        test_partition.coords,
        test_partition.energies,
        test_partition.forces,
        'kcal/mol',
        'Angstrom',
        theory='mp2.def2tzvp',
        #e_units_calc='hartree',
        #r_units_calc='bohr',
        write=False
    )

    assert test_dataset.dataset['r_unit'] == 'Angstrom'
    assert np.allclose(test_dataset.dataset['R'][0][4],
                       np.array([-1.39912, -2.017925, -0.902479]))

    assert test_dataset.dataset['e_unit'] == 'kcal/mol'
    assert isclose(test_dataset.dataset['E'][0], -143672.8876798964)
    assert isclose(test_dataset.dataset['E_max'], -143668.556633975)
    assert isclose(test_dataset.dataset['E_mean'], -143671.3661650087)
    assert isclose(test_dataset.dataset['E_min'], -143674.0760573396)
    assert isclose(test_dataset.dataset['E_var'], 1.1866207743)

    assert isclose(test_dataset.dataset['F_max'], 75.0698836347)
    assert np.allclose(test_dataset.dataset['F'][30][0],
                       np.array([6.81548275, -14.34210238, -63.49876045]))
    assert isclose(
        test_dataset.dataset['F_mean'], -2.1959649048e-08, abs_tol=0.0
    )
    assert isclose(test_dataset.dataset['F_min'], -75.0499618465)
    assert isclose(test_dataset.dataset['F_var'], 373.3360402970)

    assert test_dataset.dataset['e_unit'] == 'kcal/mol'
    assert test_dataset.dataset['name'] == '4H2O-ABC-300K-1-dataset'
    assert test_dataset.dataset['system'] == 'solvent'
    assert test_dataset.dataset['solvent'] == 'water'
    assert test_dataset.dataset['cluster_size'] == 3
    assert test_dataset.dataset['theory'] == 'mp2.def2tzvp'


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

