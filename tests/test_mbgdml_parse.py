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

import mbgdml.parse as parse
import mbgdml.utils as utils

# Must be run from mbGDML root directory.

def test_parse_parse_cluster():

    coord_path = './tests/data/md/4h2o.abc0-orca.md-mp2.def2tzvp.300k-1.traj'

    z_all, _, R_list = parse.parse_stringfile(coord_path)
    assert len(set(tuple(i) for i in z_all)) == 1
    z_elements = z_all[0]
    assert z_elements == [
        'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H'
    ]
    z = np.array(utils.atoms_by_number(z_elements))
    R = np.array(R_list)
    cluster = parse.parse_cluster(z, R[0])
    assert list(cluster.keys()) == [0, 1, 2, 3]
    assert np.allclose(cluster[0]['z'], np.array([8, 1, 1]))
    assert np.allclose(
        cluster[0]['R'],
        np.array(
            [[ 1.52130901,  1.11308001,  0.393073  ],
             [ 2.36427601,  1.14014601, -0.069582  ],
             [ 0.836238,    1.27620401, -0.29389   ]]
        )
    )
