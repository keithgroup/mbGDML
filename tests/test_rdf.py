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

"""Tests for `mbgdml.analysis.rdf`."""

from math import isclose
import pytest
import numpy as np

from mbgdml.analysis.rdf import RDF

# Must be run from mbGDML root directory.

rdf_data_dir = './tests/data/other/meoh-rdf'


def test_meoh_gr_OO():
    comp_id_pair = ('meoh', 'meoh')
    entity_idxs = (0, 0)

    bin_width = 0.05
    rdf_range = (0.0, 8.0)

    Z = np.load(f'{rdf_data_dir}/61meoh-Z.npy')
    R = np.load(f'{rdf_data_dir}/61meoh-R.npy')
    entity_ids = np.load(f'{rdf_data_dir}/61meoh-entity_ids.npy')
    comp_ids = np.load(f'{rdf_data_dir}/61meoh-comp_ids.npy')
    cell_vectors = np.array(
        [[16.0, 0.0, 0.0], [0.0, 16.0, 0.0], [0.0, 0.0, 16.0]]
    )

    rdf = RDF(
        Z, entity_ids, comp_ids, cell_vectors, bin_width=bin_width,
        rdf_range=rdf_range, inter_only=True, n_workers=None
    )
    bins, gr = rdf.run(R, comp_id_pair, entity_idxs, step=1)

    bins_ref = np.array([0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425,
        0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875,
        0.925, 0.975, 1.025, 1.075, 1.125, 1.175, 1.225, 1.275, 1.325,
        1.375, 1.425, 1.475, 1.525, 1.575, 1.625, 1.675, 1.725, 1.775,
        1.825, 1.875, 1.925, 1.975, 2.025, 2.075, 2.125, 2.175, 2.225,
        2.275, 2.325, 2.375, 2.425, 2.475, 2.525, 2.575, 2.625, 2.675,
        2.725, 2.775, 2.825, 2.875, 2.925, 2.975, 3.025, 3.075, 3.125,
        3.175, 3.225, 3.275, 3.325, 3.375, 3.425, 3.475, 3.525, 3.575,
        3.625, 3.675, 3.725, 3.775, 3.825, 3.875, 3.925, 3.975, 4.025,
        4.075, 4.125, 4.175, 4.225, 4.275, 4.325, 4.375, 4.425, 4.475,
        4.525, 4.575, 4.625, 4.675, 4.725, 4.775, 4.825, 4.875, 4.925,
        4.975, 5.025, 5.075, 5.125, 5.175, 5.225, 5.275, 5.325, 5.375,
        5.425, 5.475, 5.525, 5.575, 5.625, 5.675, 5.725, 5.775, 5.825,
        5.875, 5.925, 5.975, 6.025, 6.075, 6.125, 6.175, 6.225, 6.275,
        6.325, 6.375, 6.425, 6.475, 6.525, 6.575, 6.625, 6.675, 6.725,
        6.775, 6.825, 6.875, 6.925, 6.975, 7.025, 7.075, 7.125, 7.175,
        7.225, 7.275, 7.325, 7.375, 7.425, 7.475, 7.525, 7.575, 7.625,
        7.675, 7.725, 7.775, 7.825, 7.875, 7.925, 7.975]
    )
    gr_ref = np.array([0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.42402177, 0.17445566,
        0.        , 0.59095385, 1.80936126, 2.73798751, 2.59046395,
        2.96054209, 1.69615035, 0.64644801, 1.58215953, 1.65016999,
        1.01214257, 1.695281  , 1.6049901 , 0.14134894, 0.99325063,
        0.36533445, 0.06444178, 0.71928554, 0.30366877, 0.82598072,
        0.68804162, 0.08361632, 0.18975976, 0.42201471, 1.15526716,
        0.69991735, 1.04695588, 0.66425921, 0.7861813 , 0.51853352,
        0.28584789, 0.94388915, 1.75855134, 1.18532879, 1.21730759,
        0.77967174, 0.85696768, 0.83749199, 1.03698548, 1.77884617,
        1.0090563 , 1.22539179, 1.36557368, 0.89644636, 0.71801551,
        1.21863283, 1.00989077, 1.45394399, 0.99866931, 1.29533373,
        1.34022242, 0.84369062, 0.6916844 , 1.22375007, 1.50055157,
        1.58745892, 1.16834075, 0.88777195, 1.27091273, 0.80810285,
        1.19031425, 0.56160687, 0.78809599, 1.17246615, 1.05425819,
        0.80109416, 0.61942179, 0.77405512, 0.74075154, 1.01777272,
        1.15796225, 1.03280362, 1.04449671, 0.72869753, 0.6710746 ,
        0.84136062, 1.78978855, 0.91159036, 0.95786325, 1.24900628,
        1.15463553, 1.63979085, 1.16062307, 0.98339848, 0.65376325,
        1.07875599, 1.24654896, 1.00991883, 0.77253984, 0.94455753,
        1.25598296, 0.76147925, 1.1157184 , 0.99642798, 0.93491499,
        0.94230028, 0.82989292, 0.67459094, 1.31814929, 1.16669159,
        0.93734756, 0.82568383, 1.12124063, 0.94944615, 0.61484794,
        0.78375142, 1.21592029, 1.12011088, 1.3782731 , 1.01938343]
    )

    assert np.allclose(bins, bins_ref)
    assert np.allclose(gr, gr_ref)

