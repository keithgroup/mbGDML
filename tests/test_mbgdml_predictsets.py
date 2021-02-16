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

import mbgdml.data as data

# Must be run from mbGDML root directory.

def test_data_create_predictset():

    dataset_path = './tests/data/datasets/4H2O-2mer-dataset.npz'
    model_paths = [
        './tests/data/models/4H2O-1mer-model-MP2.def2-TZVP-train300-sym2.npz',
        './tests/data/models/4H2O-2body-model-MP2.def2-TZVP-train300-sym8.npz'
    ]

    test_predictset = data.predictSet()
    test_predictset.load_models(model_paths)
    test_predictset.load_dataset(dataset_path)

    # Reducing number of data to the first five structures
    test_predictset.R = test_predictset.dataset['R'][0:5, :, :]
    test_predictset.E = test_predictset.dataset['E'][0:5]
    test_predictset.F = test_predictset.dataset['F'][0:5, :, :]

    test_predictset.predictset

    #TODO Add assert statements
