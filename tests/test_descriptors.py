#!/usr/bin/env python
# MIT License
#
# Copyright (c) 2020-2023, Alex M. Maldonado
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

# pylint: skip-file

import numpy as np
from mbgdml import descriptors

# Must be run from mbGDML root directory.

DSET_3H2O_PATH = "./tests/data/datasets/2h2o/16h2o.yoo.etal.boat.b.2h2o-dset.mb.npz"


def load_3h2o_dset():
    return dict(np.load(DSET_3H2O_PATH, allow_pickle=True))


def test_com_distance_sum():
    dset = load_3h2o_dset()
    r_criteria = descriptors.Criteria(
        descriptors.com_distance_sum,
        desc_kwargs={"entity_ids": np.array([0, 0, 0, 1, 1, 1])},
        cutoff=5.3809148637976385,
    )
    accept_r, desc_v = r_criteria.accept(dset["z"], dset["R"][42])
    assert not accept_r
    assert desc_v == 5.3809148637976385
