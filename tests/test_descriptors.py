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

"""Tests for `mbgdml.descriptor`."""

from math import isclose
import pytest
import numpy as np

from mbgdml import descriptors

# Must be run from mbGDML root directory.

Rset_140h2o_path = './tests/data/structuresets/140h2o.sphere.gfn2.md.500k.prod1.npz'

def load_140h2o_rset():
    return dict(np.load(Rset_140h2o_path, allow_pickle=True))

def test_com_distance_sum():
    rset = load_140h2o_rset()
    r_criteria = descriptors.Criteria(
        descriptors.com_distance_sum,
        desc_kwargs={'entity_ids': rset['entity_ids']},
        cutoff=1001.506
    )
    accept_r, desc_v = r_criteria.accept(rset['z'], rset['R'][42])
    assert accept_r == False
    assert desc_v == 1001.506707631225
