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

"""Tests for `mbgdml.data.structureset`."""

from math import isclose
import pytest
import numpy as np

from mbgdml.data import structureSet

# Must be run from mbGDML root directory.

def test_structureset_from_xyz():
    coord_paths = './tests/data/md/4h2o.abc0-orca.md-mp2.def2tzvp.300k-1.traj'

    test_structureset = structureSet()
    test_structureset.read_xyz(coord_paths, 'coords', r_unit='Angstrom')
    assert np.all(
        [test_structureset.z, np.array([8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1])]
    )
    assert np.allclose(
        test_structureset.R[13],
        np.array(
            [[ 1.60356892,  1.15940522,  0.45120149],
             [ 2.44897814,  0.98409976, -0.02443958],
             [ 0.91232083,  1.31043934, -0.32129452],
             [-1.38698267, -1.18108194, -0.36717795],
             [-1.24963245, -1.95197596, -0.92292253],
             [-0.81853711, -1.22776338,  0.45900949],
             [-0.43544764,  1.15521674, -1.45105255],
             [-0.8170486 ,  0.26677825, -1.2961774 ],
             [-1.15593236,  1.75261822, -1.2450081 ],
             [ 0.48310408, -1.085288  ,  1.49429163],
             [ 0.25663241, -0.85432318,  2.40218465],
             [ 0.75853895, -0.23317651,  1.09461031]]
        )
    )
    assert test_structureset.name == 'structureset'
    assert test_structureset.r_unit == 'Angstrom'
    assert test_structureset.md5 == 'e157c5e33c7e94e7970d5cb4b3156e66'
