#!/usr/bin/env python
# MIT License
# 
# Copyright (c) 2020-2022, Alex M. Maldonado
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

"""Tests for `mbgdml.criteria`."""

from math import isclose
import pytest
import numpy as np

from mbgdml import criteria
from mbgdml.data import structureSet

# Must be run from mbGDML root directory.

Rset_140h2o_path = './tests/data/structuresets/140h2o.sphere.gfn2.md.500k.prod1.npz'

criteria_molecule_index = {'h2o': 0, 'mecn': 4, 'meoh': 0}

def load_140h2o_rset():
    return structureSet(Rset_140h2o_path)

def test_get_z_slice():
    # 140 H2O
    rset = load_140h2o_rset()
    z_slice = criteria.get_z_slice(
        rset.entity_ids, rset.comp_ids, criteria_molecule_index
    )
    z_slice_correct = np.array(
        [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51, 54,57,60,63,66,69,
        72,75,78,81,84,87,90,93,96,99,102,105,108,111,114,117,120,123,126,129,
        132,135,138,141,144,147,150,153,156,159,162,165,168,171,174,177,180,
        183,186,189,192,195,198,201,204,207,210,213,216,219,222,225,228,231,
        234,237,240,243,246,249,252,255,258,261,264,267,270,273,276,279,282,
        285,288,291,294,297,300,303,306,309,312,315,318,321,324,327,330,333,336,
        339,342,345,348,351,354,357,360,363,366,369,372,375,378,381,384,387,390,
        393,396,399,402,405,408,411,414,417]
    )
    assert np.all(z_slice == z_slice_correct)

    # H2O, MECN, and MEOH
    entity_ids = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    comp_ids = np.array(['h2o', 'mecn', 'meoh'])
    z_slice = criteria.get_z_slice(
        entity_ids, comp_ids, criteria_molecule_index
    )
    z_slice_correct = np.array([0, 7, 9])
    assert np.all(z_slice == z_slice_correct)

def test_distance_all():
    rset = load_140h2o_rset()
    z_slice = criteria.get_z_slice(
        rset.entity_ids, rset.comp_ids, criteria_molecule_index
    )
    accept_r, max_distance = criteria.distance_all(
        rset.z, rset.R[42], z_slice, rset.entity_ids, cutoff=[21.03]
    )
    assert max_distance == 21.039300581815084
    assert accept_r == False

def test_distance_sum():
    rset = load_140h2o_rset()
    z_slice = criteria.get_z_slice(
        rset.entity_ids, rset.comp_ids, criteria_molecule_index
    )
    accept_r, distance = criteria.distance_sum(
        rset.z, rset.R[42], z_slice, rset.entity_ids, cutoff=[1002.07]
    )
    assert accept_r == False
    assert distance == 1002.0768889537801

def test_cm_distance_sum():
    rset = load_140h2o_rset()
    accept_r, distance = criteria.cm_distance_sum(
        rset.z, rset.R[42], [], rset.entity_ids, cutoff=[1001.506]
    )
    assert accept_r == False
    assert distance == 1001.5068227366426
