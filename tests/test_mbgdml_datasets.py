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
from mbgdml import criteria
from mbgdml import utils

# Must be run from mbGDML root directory.

rset_path_140h2o = './tests/data/structuresets/140h2o.sphere.gfn2.md.500k.prod1.npz'
molecule_sizes = {
    'h2o': 3,
    'mecn': 6,
    'meoh': 6
}

def trim_140h2o_rset():
    """Trims the 140h2o structure set to make tests easier.
    """
    n_R = 3  # Number of structures to keep.
    n_entities = 5  # Number of molecules to keep in each structure.
    molecule_size = molecule_sizes['h2o']  # Number of atoms in a water molecule.
    rset = data.structureSet(rset_path_140h2o)

    assert rset.type == 's'
    assert rset.md5 == '8726c482c19cdf7889cd1e62b9e9c8e1'

    # Trims and checks z.
    rset.z = rset.z[:n_entities*molecule_size]
    z = np.array([8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1])
    assert np.all(rset.z == z)
    
    # Trims and checks R.
    rset.R = rset.R[:n_R, :molecule_size*n_entities]
    r_2 = np.array([
        [ 6.07124359,  0.7619846,   0.58984577],
        [ 6.47807882, -0.18138608,  0.67938893],
        [ 5.14951519,  0.76914325,  0.66198299],
        [-4.28204826, -3.57395445,  0.81850038],
        [-4.33819343, -4.29134079,  0.12722189],
        [-4.33829705, -2.80167393,  0.40818626],
        [-2.82371582, -3.52131402, -4.12086561],
        [-2.96180787, -4.46433929, -3.79287547],
        [-1.85909245, -3.46817877, -4.3649756,],
        [ 6.24586283, -1.76605224,  0.72883595],
        [ 5.51074538, -2.26847206,  1.21432844],
        [ 6.92768826, -2.3359825,   0.25592583],
        [-2.44826194, -6.14429515, -3.37660252],
        [-2.19536627, -6.12210888, -2.51171765],
        [-2.65953004, -7.04099688, -3.59504014]
    ])
    assert np.allclose(rset.R[2], r_2)
    assert rset.z.shape[0] == rset.R.shape[1]

    # Trims and checks entity_ids.
    rset.entity_ids = rset.entity_ids[:n_entities*molecule_size]
    entity_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    assert np.all(rset.entity_ids == entity_ids)

    # Trims and checks comp_ids
    rset.comp_ids = rset.comp_ids[:n_entities]
    comp_ids = np.array([
        ['0', 'h2o'], ['1', 'h2o'], ['2', 'h2o'], ['3', 'h2o'], ['4', 'h2o']
    ])
    assert np.all(rset.comp_ids == comp_ids)

    # Confirms changes with MD5.
    assert rset.md5 == 'da254c95956709d1a00512f1ac7c0bbb'

    return rset

def normal_sampling_all_2mers(dset, data):
    """Samples all dimers (2mers) from a structure or data set. All criteria is
    ignored.
    """
    quantity = 'all'
    size = 2
    selected_rset_id = None  # Always None for structure sets.
    r_criteria = None
    z_slice = np.array([])
    cutoff = []
    center_structures = False
    sampling_updates = False

    dset.sample_structures(
        data, quantity, size, selected_rset_id=selected_rset_id,
        criteria=r_criteria, z_slice=z_slice, cutoff=cutoff,
        center_structures=center_structures, sampling_updates=sampling_updates
    )
    return dset

def centered_sampling_all_2mers(dset, data):
    """Samples all dimers (2mers) from a structure or data set. All criteria is
    ignored.
    """
    quantity = 'all'
    size = 2
    selected_rset_id = None  # Always None for structure sets.
    r_criteria = None
    z_slice = np.array([])
    cutoff = []
    center_structures = True
    sampling_updates = False

    dset.sample_structures(
        data, quantity, size, selected_rset_id=selected_rset_id,
        criteria=r_criteria, z_slice=z_slice, cutoff=cutoff,
        center_structures=center_structures, sampling_updates=sampling_updates
    )
    return dset

def criteria_sampling_all_2mers(dset, data):
    """Samples all dimers (2mers) from a structure or data set. All criteria is
    ignored.
    """
    quantity = 'all'
    size = 2
    selected_rset_id = None  # Always None for structure sets.
    r_criteria = criteria.cm_distance_sum
    z_slice = np.array([])
    cutoff = [6.0]
    center_structures = True
    sampling_updates = False

    dset.sample_structures(
        data, quantity, size, selected_rset_id=selected_rset_id,
        criteria=r_criteria, z_slice=z_slice, cutoff=cutoff,
        center_structures=center_structures, sampling_updates=sampling_updates
    )
    return dset

def test_rset_sampling_all_2mers_normal():
    """Sampling all dimers (2mers) from trimmed 140h2o structure set.
    """
    rset = trim_140h2o_rset()

    ###   NORMAL SAMPLING   ###
    dset = data.dataSet()
    dset.name = '140h2o.sphere.gfn2.md.500k.prod1'
    dset = normal_sampling_all_2mers(dset, rset)

    # Checking properties.
    assert dset.Rset_md5 == {0: 'da254c95956709d1a00512f1ac7c0bbb'}
    assert dset.Rset_info.shape == (30, 4)
    assert np.all(dset.Rset_info[:, :1] == np.zeros((30,)))
    assert np.all(
        dset.Rset_info[:, 1] == np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    )
    assert dset.Rset_info.shape == np.unique(dset.Rset_info, axis=0).shape
    assert np.all(dset.entity_ids == np.array([0, 0, 0, 1, 1, 1]))
    assert np.all(dset.comp_ids == np.array([['0', 'h2o'], ['1', 'h2o']]))
    assert np.all(dset.z == np.array([8, 1, 1, 8, 1, 1]))

    # Checking R.
    assert dset.R.shape == (30, 6, 3)
    rset_info_r_check = np.array([0, 1, 1, 4])
    r_index = np.where(
        np.all(dset.Rset_info == rset_info_r_check, axis=1)
    )[0][0]
    r_check = np.array([
        [-4.27804369, -3.56574992,  0.81519167],
        [-4.3569076,  -4.2647005,   0.1558876],
        [-4.35184085, -2.82879184,  0.39925437],
        [-2.44708832, -6.14572336, -3.36929742],
        [-2.18964657, -6.13868747, -2.48473228],
        [-2.64909444, -7.04677952, -3.60878085]
    ])
    assert np.allclose(dset.R[r_index], r_check)

    # Checking E.
    assert dset.E.shape == (30,)
    assert np.all(np.isnan(dset.E))

    # Checking F
    assert dset.F.shape == (30, 6, 3)
    assert np.all(np.isnan(dset.F))

def test_rset_sampling_all_2mers_ignore_duplicate():
    rset = trim_140h2o_rset()
    dset = data.dataSet()
    dset.name = '140h2o.sphere.gfn2.md.500k.prod1'
    dset = normal_sampling_all_2mers(dset, rset)

    dset_duplicate = normal_sampling_all_2mers(dset, rset)
    assert dset_duplicate.Rset_md5 == {0: 'da254c95956709d1a00512f1ac7c0bbb'}
    assert dset_duplicate.Rset_info.shape == (30, 4)
    assert np.all(dset.entity_ids == np.array([0, 0, 0, 1, 1, 1]))
    assert np.all(dset.comp_ids == np.array([['0', 'h2o'], ['1', 'h2o']]))
    assert dset_duplicate.R.shape == (30, 6, 3)

def test_rset_sampling_all_2mers_centering():
    rset = trim_140h2o_rset()

    dset = data.dataSet()
    dset.name = '140h2o.sphere.gfn2.md.500k.prod1'
    dset = normal_sampling_all_2mers(dset, rset)
    centered_R = dset._center_structures(dset.z, dset.R)

    dset_centered = data.dataSet()
    dset_centered.name = '140h2o.sphere.gfn2.md.500k.prod1-centered'
    dset_centered = centered_sampling_all_2mers(dset_centered, rset)

    assert np.allclose(centered_R, dset_centered.R)

def test_rset_sampling_all_2mers_criteria():
    rset = trim_140h2o_rset()

    dset_centered = data.dataSet()
    dset_centered.name = '140h2o.sphere.gfn2.md.500k.prod1-centered'
    dset_centered = centered_sampling_all_2mers(dset_centered, rset)

    dset_criteria = data.dataSet()
    dset_criteria.name = '140h2o.sphere.gfn2.md.500k.prod1-criteria'
    dset_criteria = criteria_sampling_all_2mers(dset_criteria, rset)

    Rset_info_accpetable_criteria = np.array([
        [0,0,0,3], [0,0,1,2], [0,0,1,4], [0,0,2,4], [0,1,0,3], [0,1,1,2],
        [0,1,1,4], [0,1,2,4], [0,2,0,3], [0,2,1,2], [0,2,1,4], [0,2,2,4]
    ])
    
    assert np.array_equal(dset_criteria.Rset_info, Rset_info_accpetable_criteria)

def centered_rset_sampling_num_2mers_criteria(dset, data):
    """Samples five all dimers (2mers) from a structure or data set.
    Criteria is not ignored.
    """
    quantity = 5
    size = 2
    selected_rset_id = None  # Always None for structure sets.
    r_criteria = criteria.cm_distance_sum
    z_slice = np.array([])
    cutoff = [6.0]
    center_structures = True
    sampling_updates = False

    dset.sample_structures(
        data, quantity, size, selected_rset_id=selected_rset_id,
        criteria=r_criteria, z_slice=z_slice, cutoff=cutoff,
        center_structures=center_structures, sampling_updates=sampling_updates
    )
    return dset

def test_dset_default_attributes():
    dset = data.dataSet()

    assert isinstance(dset.Rset_md5, dict)
    assert len(dset.Rset_md5) == 0
    assert dset.Rset_info.shape == (1, 0)

    assert dset.criteria == ''
    assert dset.z_slice.shape == (0,)
    assert dset.cutoff.shape == (0,)

    assert dset.z.shape == (0,)
    assert dset.R.shape == (1, 1, 0)
    assert dset.E.shape == (0,)
    assert dset.F.shape == (1, 1, 0)

    assert dset.entity_ids.shape == (0,)
    assert dset.comp_ids.shape == (1, 0)

    try:
        dset.md5
    except AttributeError:
        pass

def test_rset_sampling_num_2mers_criteria():
    rset = trim_140h2o_rset()

    dset = data.dataSet()
    dset.name = '140h2o.sphere.gfn2.md.500k.prod1'
    dset = centered_rset_sampling_num_2mers_criteria(dset, rset)
    
    assert isinstance(dset.criteria, str)
    assert dset.criteria in criteria.__dict__
    assert dset.z_slice.shape == (0,)
    assert dset.cutoff.shape == (1,)
    assert np.array_equal(dset.cutoff, np.array([6.]))

    assert dset.r_unit == 'Angstrom'
    assert np.array_equal(dset.z, np.array([8, 1, 1, 8, 1, 1]))
    assert dset.R.shape == (5, 6, 3)
    assert dset.E.shape == (5,)
    assert dset.F.shape == (5, 6, 3)

    assert dset.Rset_md5 == {0: 'da254c95956709d1a00512f1ac7c0bbb'}
    assert np.array_equal(dset.entity_ids, np.array([0, 0, 0, 1, 1, 1]))
    assert np.array_equal(dset.comp_ids, np.array([['0', 'h2o'], ['1', 'h2o']]))

    # Checks that Rset_info is correct.
    for i in range(len(dset.R)):
        r_dset = dset.R[i]

        r_info = dset.Rset_info[i]
        i_struct = r_info[1]
        i_entity_ids = r_info[2:]
        atom_idx = utils.get_R_slice(i_entity_ids, rset.entity_ids)
        r_rset = rset.R[i_struct][atom_idx]
        r_rset_centered = dset._center_structures(dset.z, r_rset)

        assert np.allclose(r_dset, r_rset_centered)


def test_rset_sampling_num_2mers_additional():
    rset = trim_140h2o_rset()

    dset = data.dataSet()
    dset.name = '140h2o.sphere.gfn2.md.500k.prod1'
    dset = centered_rset_sampling_num_2mers_criteria(dset, rset)

    # Ensure energies and forces are not overwritten
    i_test = 1
    e_test = -47583.29857
    dset.E[i_test] = e_test
    f_test = np.array([
        [4.4, 2.8, 6.0],
        [-3.65, 34.0, 2.3],
        [4.4, 2.8, 6.0],
        [-3.65, 34.0, 2.3],
        [4.4, 2.8, 6.0],
        [-3.65, 34.0, 2.3],
    ])
    dset.F[i_test] = f_test

    dset = centered_rset_sampling_num_2mers_criteria(dset, rset)

    assert dset.Rset_md5 == {0: 'da254c95956709d1a00512f1ac7c0bbb'}
    assert np.array_equal(dset.entity_ids, np.array([0, 0, 0, 1, 1, 1]))
    assert np.array_equal(dset.comp_ids, np.array([['0', 'h2o'], ['1', 'h2o']]))

    assert np.array_equal(dset.z, np.array([8, 1, 1, 8, 1, 1]))
    assert dset.R.shape == (10, 6, 3)
    assert dset.E.shape == (10,)
    assert np.allclose(dset.E[i_test], e_test)
    assert dset.F.shape == (10, 6, 3)
    assert np.allclose(dset.F[i_test], f_test)

    # Checks that Rset_info is correct.
    for i in range(len(dset.R)):
        r_dset = dset.R[i]

        r_info = dset.Rset_info[i]
        i_struct = r_info[1]
        i_entity_ids = r_info[2:]
        atom_idx = utils.get_R_slice(i_entity_ids, rset.entity_ids)
        r_rset = rset.R[i_struct][atom_idx]
        r_rset_centered = dset._center_structures(dset.z, r_rset)

        assert np.allclose(r_dset, r_rset_centered)

    print(dset.E)

"""
def test_mbdataset():
    dataset_2mer_path = './tests/data/datasets/4H2O-2mer-dataset.npz'
    model_1mer_path = './tests/data/models/4H2O-1mer-model-MP2.def2-TZVP-train300-sym2.npz'
    
    dataset = data.dataSet(dataset_2mer_path)
    mb_dataset = data.dataSet()
    mb_dataset.create_mb(dataset, [model_1mer_path])

    assert mb_dataset.mb == 2
    assert mb_dataset.system_info['system'] == 'solvent'
    assert mb_dataset.system_info['solvent_name'] == 'water'
    assert np.allclose(
        mb_dataset.z,
        np.array([8, 1, 1, 8, 1, 1])
    )
    assert np.allclose(
        mb_dataset.R[23],
        np.array(
            [[ 1.63125 ,  1.23959 ,  0.426174],
             [ 2.346105,  1.075848, -0.216122],
             [ 0.855357,  1.299862, -0.15629 ],
             [-1.449062, -1.07853 , -0.450404],
             [-1.366435, -1.86773 , -0.997238],
             [-0.777412, -1.236662,  0.247103]]
        )
    )
    assert mb_dataset.E[23] == -1.884227499962435
    assert np.allclose(np.array(mb_dataset.E_var), np.array(2.52854648))
    assert np.allclose(np.array(mb_dataset.F_var), np.array(12.59440808))
"""
