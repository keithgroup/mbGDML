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
    ref_dataset_path = ''.join([test_path, 'data/ABC-4MeOH-300K-1-gdml.npz'])
    ref_output_path = ''.join([test_path, 'data/out-4MeOH-300K-1-ABC.out'])

    ref_dataset = np.load(ref_dataset_path)
    test_partition = mbgdml.data.PartitionCalcOutput(ref_output_path)
    test_partition.create_dataset()
    
    assert np.array_equal(ref_dataset.f.E, test_partition.dataset['E'])