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

"""Tests for `mbgdml` package."""

import pytest
import numpy as np

import mbgdml.data as data
from mbgdml import criteria
from mbgdml import utils

# Must be run from mbGDML root directory.

rset_path_140h2o = './tests/data/structuresets/140h2o.sphere.gfn2.md.500k.prod1.npz'
dset_dir = './tests/data/datasets'
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
    comp_ids = np.array(['h2o', 'h2o', 'h2o', 'h2o', 'h2o'])
    assert np.all(rset.comp_ids == comp_ids)

    # Confirms changes with MD5.
    assert rset.md5 == 'da254c95956709d1a00512f1ac7c0bbb'

    return rset

def dset_sample_structures(
    dset, data, quantity, size, r_criteria,
    z_slice, cutoff, center_structures, sampling_updates
):
    """Generic sampling function.
    """
    dset.sample_structures(
        data, quantity, size,
        criteria=r_criteria, z_slice=z_slice, cutoff=cutoff,
        center_structures=center_structures, sampling_updates=sampling_updates
    )
    return dset

def check_R_with_rset(dset, rset, centered):
    """Uses structure information from r_prov_specs to check structure coordinates.

    Parameters
    ----------
    dset : :obj:`mbgdml.data.dataset.dataSet`
        The data set.
    rset : :obj:`mbgdml.data.structureset.structureSet`
        The structure set.
    centered : :obj:`bool`
        If the dset coordinates were centered with respect to the cluster's
        center of mass.
    """
    z_dset = dset.z
    R_dset = dset.R
    r_prov_specs = dset.r_prov_specs
    R_rset = rset.R
    rset_entity_ids = rset.entity_ids
    for i_r_dset in range(len(dset.R)):
        r_dset = R_dset[i_r_dset]

        r_prov_spec = r_prov_specs[i_r_dset]
        i_r_rset = r_prov_spec[1]
        r_rset_entity_ids = r_prov_spec[2:]
        r_slice_rset = utils.get_R_slice(r_rset_entity_ids, rset_entity_ids)
        r_rset = R_rset[i_r_rset][r_slice_rset]

        if centered == True:
            r_rset = utils.center_structures(z_dset, r_rset)
        
        assert np.allclose(r_dset, r_rset, atol=5.1e-07, rtol=0)

def test_rset_sampling_all_2mers_normal():
    """Sampling all dimers (2mers) from trimmed 140h2o structure set.
    """
    rset = trim_140h2o_rset()

    ###   NORMAL SAMPLING   ###
    dset = data.dataSet()
    dset.name = '140h2o.sphere.gfn2.md.500k.prod1'
    dset = dset_sample_structures(
        dset, rset, 'all', 2, None,
        np.array([]), np.array([]), False, False
    )

    # Checking properties.
    assert dset.r_prov_ids == {0: 'da254c95956709d1a00512f1ac7c0bbb'}
    assert dset.r_prov_specs.shape == (30, 4)
    assert np.all(dset.r_prov_specs[:, :1] == np.zeros((30,)))
    assert np.all(
        dset.r_prov_specs[:, 1] == np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    )
    assert dset.r_prov_specs.shape == np.unique(dset.r_prov_specs, axis=0).shape
    assert np.all(dset.entity_ids == np.array([0, 0, 0, 1, 1, 1]))
    assert np.all(dset.comp_ids == np.array(['h2o', 'h2o']))
    assert np.all(dset.z == np.array([8, 1, 1, 8, 1, 1]))

    # Checking R.
    assert dset.R.shape == (30, 6, 3)
    r_prov_specs_r_check = np.array([0, 1, 1, 4])
    r_index = np.where(
        np.all(dset.r_prov_specs == r_prov_specs_r_check, axis=1)
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
    dset = dset_sample_structures(
        dset, rset, 'all', 2, None,
        np.array([]), np.array([]), False, False
    )

    dset_duplicate = dset_sample_structures(
        dset, rset, 'all', 2, None,
        np.array([]), np.array([]), False, False
    )
    assert dset_duplicate.r_prov_ids == {0: 'da254c95956709d1a00512f1ac7c0bbb'}
    assert dset_duplicate.r_prov_specs.shape == (30, 4)
    assert np.all(dset.entity_ids == np.array([0, 0, 0, 1, 1, 1]))
    assert np.all(dset.comp_ids == np.array(['h2o', 'h2o']))
    assert dset_duplicate.R.shape == (30, 6, 3)

def test_rset_sampling_all_2mers_centering():
    rset = trim_140h2o_rset()

    dset = data.dataSet()
    dset.name = '140h2o.sphere.gfn2.md.500k.prod1'
    dset = dset_sample_structures(
        dset, rset, 'all', 2, None,
        np.array([]), np.array([]), False, False
    )
    centered_R = utils.center_structures(dset.z, dset.R)

    dset_centered = data.dataSet()
    dset_centered.name = '140h2o.sphere.gfn2.md.500k.prod1-centered'
    dset_centered = dset_sample_structures(
        dset_centered, rset, 'all', 2, None,
        np.array([]), np.array([]), True, False
    )

    assert np.allclose(centered_R, dset_centered.R)

def test_rset_sampling_all_2mers_criteria():
    rset = trim_140h2o_rset()

    dset_centered = data.dataSet()
    dset_centered.name = '140h2o.sphere.gfn2.md.500k.prod1-centered'
    dset_centered = dset_sample_structures(
        dset_centered, rset, 'all', 2, None,
        np.array([]), np.array([]), True, False
    )

    dset_criteria = data.dataSet()
    dset_criteria.name = '140h2o.sphere.gfn2.md.500k.prod1-criteria'
    dset_criteria = dset_sample_structures(
        dset_criteria, rset, 'all', 2, criteria.cm_distance_sum,
        np.array([]), np.array([6.0]), True, False
    )

    r_prov_specs_accpetable_criteria = np.array([
        [0,0,0,3], [0,0,1,2], [0,0,1,4], [0,0,2,4], [0,1,0,3], [0,1,1,2],
        [0,1,1,4], [0,1,2,4], [0,2,0,3], [0,2,1,2], [0,2,1,4], [0,2,2,4]
    ])
    
    assert np.array_equal(dset_criteria.r_prov_specs, r_prov_specs_accpetable_criteria)

def test_dset_default_attributes():
    dset = data.dataSet()

    assert isinstance(dset.r_prov_ids, dict)
    assert len(dset.r_prov_ids) == 0
    assert dset.r_prov_specs.shape == (1, 0)

    assert dset.criteria == ''
    assert dset.z_slice.shape == (0,)
    assert dset.cutoff.shape == (0,)

    assert dset.z.shape == (0,)
    assert dset.R.shape == (1, 1, 0)
    assert dset.E.shape == (0,)
    assert dset.F.shape == (1, 1, 0)

    assert dset.entity_ids.shape == (0,)
    assert dset.comp_ids.shape == (0,)

    try:
        dset.md5
    except AttributeError:
        pass

def test_rset_sampling_num_2mers_criteria():
    rset = trim_140h2o_rset()

    dset = data.dataSet()
    dset.name = '140h2o.sphere.gfn2.md.500k.prod1'
    dset = dset_sample_structures(
        dset, rset, 5, 2, criteria.cm_distance_sum,
        np.array([]), np.array([6.0]), True, False
    )
    
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

    assert dset.r_prov_ids == {0: 'da254c95956709d1a00512f1ac7c0bbb'}
    assert np.array_equal(dset.entity_ids, np.array([0, 0, 0, 1, 1, 1]))
    assert np.array_equal(dset.comp_ids, np.array(['h2o', 'h2o']))

    check_R_with_rset(dset, rset, True)

def test_rset_sampling_num_2mers_additional():
    rset = trim_140h2o_rset()

    dset = data.dataSet()
    dset.name = '140h2o.sphere.gfn2.md.500k.prod1'
    dset = dset_sample_structures(
        dset, rset, 5, 2, criteria.cm_distance_sum,
        np.array([]), np.array([6.0]), True, False
    )

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

    dset = dset_sample_structures(
        dset, rset, 5, 2, criteria.cm_distance_sum,
        np.array([]), np.array([6.0]), True, False
    )

    assert dset.r_prov_ids == {0: 'da254c95956709d1a00512f1ac7c0bbb'}
    assert np.array_equal(dset.entity_ids, np.array([0, 0, 0, 1, 1, 1]))
    assert np.array_equal(dset.comp_ids, np.array(['h2o', 'h2o']))

    assert np.array_equal(dset.z, np.array([8, 1, 1, 8, 1, 1]))
    assert dset.R.shape == (10, 6, 3)
    assert dset.E.shape == (10,)
    assert np.allclose(dset.E[i_test], e_test)
    assert dset.F.shape == (10, 6, 3)
    assert np.allclose(dset.F[i_test], f_test)

    check_R_with_rset(dset, rset, True)

def test_dset_sampling_all_2mers_after_3mers():
    rset = trim_140h2o_rset()

    dset = data.dataSet()
    dset.name = '140h2o.sphere.gfn2.md.500k.prod1'
    dset = dset_sample_structures(
        dset, rset, 'all', 3, None,
        np.array([]), np.array([]), True, False
    )

    dset_from_dset = data.dataSet()
    dset_from_dset = dset_sample_structures(
        dset_from_dset, dset, 'all', 2, criteria.cm_distance_sum,
        np.array([]), np.array([6.0]), True, False
    )

    assert np.array_equal(dset_from_dset.entity_ids, np.array([0, 0, 0, 1, 1, 1]))
    assert np.array_equal(
        dset_from_dset.comp_ids, np.array(['h2o', 'h2o'])
    )
    assert dset_from_dset.r_prov_ids == {0: 'da254c95956709d1a00512f1ac7c0bbb'}

    assert dset_from_dset.r_prov_specs.shape == (12, 4)
    # Same as test_rset_sampling_all_2mers_criteria, but organized to match
    # the 3mer then 2mer sampling.
    r_prov_specs_accpetable_criteria = np.array([
        [0,0,1,2], [0,0,0,3], [0,0,1,4], [0,0,2,4], [0,1,1,2], [0,1,0,3],
        [0,1,1,4], [0,1,2,4], [0,2,1,2], [0,2,0,3], [0,2,1,4], [0,2,2,4]
    ])
    assert np.array_equal(dset_from_dset.r_prov_specs, r_prov_specs_accpetable_criteria)
    
    assert dset_from_dset.R.shape == (12, 6, 3)
    assert dset_from_dset.E.shape == (12,)
    assert dset_from_dset.F.shape == (12, 6, 3)

    assert dset_from_dset.criteria == 'cm_distance_sum'
    assert np.array_equal(dset_from_dset.cutoff, np.array([6.0]))

def test_sample_dset_same_size():
    """
    """
    dset_h2o_2body_path = f'{dset_dir}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.2h2o-dset.mb.npz'

    dset_h2o_2body = data.dataSet(dset_h2o_2body_path)
    
    # Trim dset_h2o_2body to 50 structures
    remaining = 50
    for key in ['r_prov_specs', 'E', 'R', 'F']:
        setattr(dset_h2o_2body, key, getattr(dset_h2o_2body, key)[:remaining])

    dset_h2o_2body_cm_6 = data.dataSet()
    dset_h2o_2body_cm_6.name = '140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.2h2o-dset.mb-cm.6'
    dset_h2o_2body_cm_6 = dset_sample_structures(
        dset_h2o_2body_cm_6, dset_h2o_2body, 'all', 2, criteria.cm_distance_sum,
        np.array([]), np.array([6.0]), True, False
    )

    assert dset_h2o_2body_cm_6.theory == 'mp2.def2tzvp.frozencore'
    assert dset_h2o_2body_cm_6.criteria == 'cm_distance_sum'
    assert np.array_equal(dset_h2o_2body_cm_6.z_slice, np.array([]))
    assert np.array_equal(dset_h2o_2body_cm_6.cutoff, np.array([6.0]))
    assert np.array_equal(dset_h2o_2body_cm_6.entity_ids, np.array([0, 0, 0, 1, 1, 1]))
    assert np.array_equal(
        dset_h2o_2body_cm_6.comp_ids, np.array(['h2o', 'h2o'])
    )
    assert dset_h2o_2body_cm_6.centered == True
    assert dset_h2o_2body_cm_6.r_unit == 'Angstrom'
    # 8726c482c19cdf7889cd1e62b9e9c8e1 is the MD5 has for the full 140h2o rset.
    assert dset_h2o_2body_cm_6.r_prov_ids == {0: '8726c482c19cdf7889cd1e62b9e9c8e1'}

    assert np.array_equal(dset_h2o_2body_cm_6.z, np.array([8, 1, 1, 8, 1, 1]))
    rset = data.structureSet(rset_path_140h2o)
    check_R_with_rset(dset_h2o_2body_cm_6, rset, True)

    # Checking energies and forces.
    dset_r_prov_specs = dset_h2o_2body_cm_6.r_prov_specs
    dset_E = dset_h2o_2body_cm_6.E
    dset_F = dset_h2o_2body_cm_6.F
    dset_sample_r_prov_specs = dset_h2o_2body.r_prov_specs
    dset_sample_E = dset_h2o_2body.E
    dset_sample_F = dset_h2o_2body.F
    for i_r in range(len(dset_h2o_2body_cm_6.R)):
        i_r_dset_sample = np.where(
            np.all(dset_sample_r_prov_specs == dset_r_prov_specs[i_r], axis=1)
        )[0][0]
        assert np.allclose(dset_E[i_r], dset_sample_E[i_r_dset_sample])
        assert np.allclose(dset_F[i_r], dset_sample_F[i_r_dset_sample])

def test_sample_dset_1mers_multiple_rsets():
    """
    """
    dset_4h2o_lit_path = f'{dset_dir}/4h2o/4h2o.temelso.etal-dset.npz'

    dset_4h2o_lit_dset = data.dataSet(dset_4h2o_lit_path)
    
    # Sample all 1mers
    dset_1mers = data.dataSet()
    dset_1mers = dset_sample_structures(
        dset_1mers, dset_4h2o_lit_dset, 'all', 1, None,
        np.array([]), np.array([]), True, False
    )

    # Checking data set
    r_prov_specs = np.array([
        [0,0,0], [0,0,1], [0,0,2], [0,0,3], [1,0,0], [1,0,1], [1,0,2], [1,0,3],
        [2,0,0], [2,0,1], [2,0,2], [2,0,3]
    ])
    assert np.array_equal(dset_1mers.r_prov_specs, r_prov_specs)
    r_prov_ids = {0: '92dd31a90a3d2a443023d9d708010a4f', 1: '5593ef822ede64f6011ece82d6702ff9', 2: '33098027b401c38efcb5f05fa33c93ad'}
    assert dset_1mers.r_prov_ids == r_prov_ids
    assert np.array_equal(dset_1mers.entity_ids, np.array([0, 0, 0]))
    assert np.array_equal(dset_1mers.comp_ids, np.array(['h2o']))
    assert dset_1mers.centered == True
    assert dset_1mers.r_unit == 'Angstrom'
    assert np.array_equal(dset_1mers.z, np.array([8, 1, 1]))

    assert dset_1mers.R.shape == (12, 3, 3)
    r_3 = np.array([
        [-0.02947763, -0.0325826, -0.05004315],
        [ 0.93292237, 0.1104174, 0.10365685],
        [-0.46497763, 0.4068174, 0.69075685]
    ])
    assert np.allclose(dset_1mers.R[3], r_3)
    assert dset_1mers.E.shape == (12,)
    for e in dset_1mers.E:
        assert np.isnan(e)
    assert dset_1mers.F.shape == (12, 3, 3)
    for f in dset_1mers.F.flatten():
        assert np.isnan(f)

def test_adding_pes_data_with_qcjson():
    dset = data.dataSet(f'{dset_dir}/6h2o/6h2o.temelso.etal-dset-no.data.npz')
    dset_ref = data.dataSet(f'{dset_dir}/6h2o/6h2o.temelso.etal-dset.npz')

    dset.add_pes_data(
        './tests/data/engrads/h2o/6h2o/6h2o.temelso.etal',
        'MP2/def2-TZVP', 'kcal/mol', 'hartree', allow_remaining_nan=False
    )
    assert np.array_equal(dset_ref.E, dset.E)
    assert np.array_equal(dset_ref.F, dset.F)
