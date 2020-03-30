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
import mbgdml

test_path = mbgdml.utils.norm_path(
    os.path.dirname(os.path.realpath(__file__))
)

def test_data_create_dataset():
    ref_output_path = ''.join([test_path, 'data/out-4MeOH-300K-1-ABC.out'])
    write_path = ''.join([test_path, 'data/'])

    test_partition = mbgdml.data.PartitionCalcOutput(ref_output_path)
    test_partition.create_dataset(write_path, 'bohr', 'hartree',
                                  theory='MP2/def2-TZVP', write=False)

    assert test_partition.dataset['r_unit'] == 'Angstrom'
    assert np.allclose(test_partition.dataset['R'][0][4],
                       np.array([1.911664, -2.195424, -0.704814]))

    assert test_partition.dataset['e_unit'] == 'kcal/mol'
    assert test_partition.dataset['E'][0] == -217458.27068287216
    assert test_partition.dataset['E_max'] == -217448.27742004013
    assert test_partition.dataset['E_mean'] == -217453.4435426626
    assert test_partition.dataset['E_min'] == -217458.27068287216
    assert test_partition.dataset['E_var'] == 3.9385025745597186

    assert test_partition.dataset['F_max'] == 65.08310037407716
    assert np.allclose(test_partition.dataset['F'][0][0],
                       np.array([-22.73440695, -11.93017795,   1.67021709]))
    assert test_partition.dataset['F_mean'] == -4.391929749921794e-09
    assert test_partition.dataset['F_min'] == -77.57149172033839
    assert test_partition.dataset['F_var'] == 216.70599395856985

    assert test_partition.dataset['cluster_size'] == 3
    assert test_partition.dataset['e_unit'] == 'kcal/mol'
    assert test_partition.dataset['name'] == 'ABC-4MeOH-300K-1-dataset'
    assert test_partition.dataset['system'] == 'solvent'
    assert test_partition.dataset['solvent'] == 'methanol'
    assert test_partition.dataset['cluster_size'] == 3
    assert test_partition.dataset['theory'] == 'MP2/def2-TZVP'