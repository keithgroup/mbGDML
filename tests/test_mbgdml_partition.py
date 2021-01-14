#!/usr/bin/env python
# MIT License
# 
# Copyright (c) 2020-2021, Alex M. Maldonado
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
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

import mbgdml.partition as partition

# Must be run from mbGDML root directory.

def test_partition_stringfile():
    coord_path = './tests/data/md/4h2o.abc0-orca.md-mp2.def2tzvp.300k-1.traj'
    test_partitions = partition.partition_stringfile(coord_path)

    assert list(test_partitions.keys()) == [
        '0', '1', '2', '3', '0,1', '0,2', '0,3', '1,2', '1,3', '2,3', '0,1,2',
        '0,1,3', '0,2,3', '1,2,3', '0,1,2,3'
    ]
    test_partition_1 = test_partitions['2']
    assert list(test_partition_1.keys()) == [
        'solvent_label', 'cluster_size', 'partition_label', 'partition_size',
        'z', 'R'
    ]
    assert np.allclose(test_partition_1['z'], np.array([8, 1, 1]))
    assert test_partition_1['R'].shape[0] == 101
    assert test_partition_1['R'].shape[2] == 3
    assert np.allclose(
        test_partition_1['R'][2],
        np.array(
            [[-0.48381516,  1.17384211, -1.4413092 ],
             [-0.90248552,  0.33071306, -1.24479905],
             [-1.21198585,  1.83409853, -1.4187445 ]]
        )
    )

    test_partition_3 = test_partitions['0,1,2']
    assert list(test_partition_1.keys()) == list(test_partition_3.keys())
    assert test_partition_3['R'].shape[0] == 101
    assert test_partition_3['R'].shape[2] == 3
    assert np.allclose(
        test_partition_3['R'][2],
        np.array(
            [[ 1.53804814,  1.11857593,  0.40316032],
             [ 2.35482839,  1.1215564,  -0.05145675],
             [ 0.80073022,  1.28633895, -0.3291415 ],
             [-1.4580503,  -1.16136864, -0.43897336],
             [-1.3726833,  -2.02344751, -0.89453557],
             [-0.83691991, -1.2963126,   0.3429272 ],
             [-0.48381516,  1.17384211, -1.4413092 ],
             [-0.90248552,  0.33071306, -1.24479905],
             [-1.21198585,  1.83409853, -1.4187445 ]]
        )
    )
