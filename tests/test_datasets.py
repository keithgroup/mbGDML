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

"""Tests for `mbgdml` package."""

# pylint: skip-file

from mbgdml import data


def test_dset_default_attributes():
    dset = data.DataSet()

    assert isinstance(dset.r_prov_ids, dict)
    assert len(dset.r_prov_ids) == 0
    assert dset.r_prov_specs.shape == (1, 0)

    assert dset.Z.shape == (0,)
    assert dset.R.shape == (1, 1, 0)
    assert dset.E.shape == (0,)
    assert dset.F.shape == (1, 1, 0)

    assert dset.entity_ids.shape == (0,)
    assert dset.comp_ids.shape == (0,)

    try:
        dset.md5
    except AttributeError:
        pass
