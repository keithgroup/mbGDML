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

"""Tests for GDML training"""

import pytest
import numpy as np
import os
from mbgdml.data import dataSet
from mbgdml._gdml.train import GDMLTrain, get_test_idxs
from mbgdml.train import mbGDMLTrain
from mbgdml.analysis.problematic import prob_structures
from mbgdml.predict import gdmlModel, predict_gdml

dset_dir = './tests/data/datasets'
train_dir = './tests/data/train'

def test_train_results_1h2o():
    """Checks the results of a training task."""
    global glob
    if 'glob' in globals():
        del glob
    
    dset_path = os.path.join(
        dset_dir, '1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-dset.npz'
    )
    dset = dataSet(dset_path)
    dset_dict = dset.asdict()

    train_dir_1h2o = os.path.join(train_dir, '1h2o/')
    train_idxs_path = os.path.join(train_dir_1h2o, 'train_idxs.npy')
    valid_idxs_path = os.path.join(train_dir_1h2o, 'valid_idxs.npy')
    train_idxs = np.load(train_idxs_path, allow_pickle=True)
    valid_idxs = np.load(valid_idxs_path, allow_pickle=True)

    n_train = 50
    n_valid = 100
    sigma = 42

    train = GDMLTrain()
    task = train.create_task(
        dset_dict,
        n_train,
        dset_dict,
        n_valid,
        sigma,
        lam=1e-15,
        use_sym=True,
        use_E=True,
        use_E_cstr=False,
        use_cprsn=False,
        solver='analytic',
        solver_tol=1e-4,
        idxs_train=train_idxs,
        idxs_valid=valid_idxs,
    )
    model = train.train(task)

    alphas_F = model['alphas_F']
    R_desc = model['R_desc']
    tril_perms_lin = model['tril_perms_lin']

    # Reference data
    alphas_F_ref = np.load(
        os.path.join(train_dir_1h2o, 'alphas_F.npy'),
        allow_pickle=True
    )
    R_desc_ref = np.load(
        os.path.join(train_dir_1h2o, 'R_desc.npy'),
        allow_pickle=True
    )
    tril_perms_lin_ref = np.load(
        os.path.join(train_dir_1h2o, 'tril_perms_lin.npy'),
        allow_pickle=True
    )

    del train

    # Coefficients will not be exactly the same.
    assert np.allclose(R_desc, R_desc_ref, rtol=1e-05, atol=1e-08)
    assert np.allclose(alphas_F, alphas_F_ref, rtol=1e2, atol=0)
    assert np.allclose(
        np.array(model['c']), np.array(331288.48632617114)
    )
    assert np.allclose(
        np.array(model['norm_y_train']), np.array(321987215081.7051),
        rtol=1e-3, atol=0
    )

def test_1h2o_train_grid_search():
    global glob
    if 'glob' in globals():
        del glob
    
    dset_path = os.path.join(
        dset_dir, '1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-dset.npz'
    )
    dset = dataSet(dset_path)

    train_dir_1h2o = os.path.join(train_dir, '1h2o/')
    train_idxs_path = os.path.join(train_dir_1h2o, 'train_idxs.npy')
    valid_idxs_path = os.path.join(train_dir_1h2o, 'valid_idxs.npy')
    train_idxs = np.load(train_idxs_path, allow_pickle=True)
    valid_idxs = np.load(valid_idxs_path, allow_pickle=True)

    n_train = 50
    n_valid = 100
    sigmas = [32, 42, 52]

    train = mbGDMLTrain(
        use_sym=True, use_E=True, use_E_cstr=False, use_cprsn=False,
        solver='analytic', lam=1e-15, solver_tol=1e-4
    )
    model = train.grid_search(
        dset,
        '1h2o',
        n_train,
        n_valid,
        sigmas,
        train_idxs=train_idxs,
        valid_idxs=valid_idxs,
        write_json=True,
        write_idxs=True,
        overwrite=True,
        save_dir='./tests/tmp/1h2o-grid'
    )

    del train

    assert model['sig'].item() == 42
    assert np.allclose(
        np.array(model['f_err'].item()['rmse']), 0.4673520776718695,
        rtol=1e-05, atol=1e-08
    )
    assert model['perms'].shape[0] == 2

def test_1h2o_train_bayes_opt():
    try:
        import bayes_opt
    except ImportError:
        pytest.skip("bayesian-optimization package not installed")
    
    global glob
    if 'glob' in globals():
        del glob
    
    dset_path = os.path.join(
        dset_dir, '1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-dset.npz'
    )
    dset = dataSet(dset_path)

    train_dir_1h2o = os.path.join(train_dir, '1h2o/')
    train_idxs_path = os.path.join(train_dir_1h2o, 'train_idxs.npy')
    valid_idxs_path = os.path.join(train_dir_1h2o, 'valid_idxs.npy')
    train_idxs = np.load(train_idxs_path, allow_pickle=True)
    valid_idxs = np.load(valid_idxs_path, allow_pickle=True)

    n_train = 50
    n_valid = 100
    sigmas = [32, 42, 52]

    train = mbGDMLTrain(
        use_sym=True, use_E=True, use_E_cstr=False, use_cprsn=False,
        solver='analytic', lam=1e-15, solver_tol=1e-4
    )
    gp_params = {'init_points': 5, 'n_iter': 5, 'alpha': 0.001}
    model, optimizer = train.bayes_opt(
        dset,
        '1h2o',
        n_train,
        n_valid,
        sigma_bounds=(2, 100),
        save_dir='./tests/tmp/1h2o-bo',
        gp_params=gp_params,
        train_idxs=train_idxs,
        valid_idxs=valid_idxs,
        overwrite=True,
        write_json=True,
        write_idxs=True,
    )

    best_sig = model['sig'].item()
    assert 37 <= best_sig <= 53
    assert model['perms'].shape[0] == 2

    del train

def test_1h2o_prob_indices():
    global glob
    if 'glob' in globals():
        del glob
    
    dset_path = os.path.join(
        dset_dir, '1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-dset.npz'
    )
    model_path = os.path.join(
        './tests/data/models', '1h2o-model.npz'
    )
    model = dict(np.load(model_path, allow_pickle=True))
    model = gdmlModel(
        model, criteria_desc_func=None,
        criteria_cutoff=None
    )
    dset = dataSet(dset_path)

    prob_s = prob_structures([model], predict_gdml)
    n_find = 100
    prob_idxs = prob_s.find(dset, n_find, save_dir='./tests/tmp')
    prob_idxs = np.sort(prob_idxs)

    ref = np.array(
        [ 
            465,   541,   653,   798,   807,   921,   953,  1058,  1240,
            1421,  1430,  1510,  1618,  1663,  1665,  1676,  1890,  2090,
            2123,  2218,  2246,  2665,  2944,  3171,  3225,  3485,  3510,
            3738,  3795,  3970,  3994,  4272,  4660,  5102,  5150,  5195,
            5230,  6394,  6471,  6787,  6900,  6961,  6986,  7257,  7725,
            7735,  7812,  7815,  8006,  8074,  8253,  8489,  8532,  8810,
            9169,  9221,  9226,  9667,  9668,  9728,  9747,  9919,  9952,
            9995, 10025, 10057, 10062, 10144, 10252, 10525, 10763, 10982,
            11005, 11012, 11024, 11404, 11730, 11745, 11747, 11864, 11970,
            12049, 12167, 12329, 12465, 12478, 12638, 12645, 12655, 12664,
            12775, 12878, 13062, 13151, 13192, 13320, 13343, 13546, 13676,
            13963
        ]
    )

    assert len(prob_idxs) == 100
    # This is a very bad test, but will work for now?
    assert len(np.setdiff1d(prob_idxs, ref)) < 20

def test_getting_test_idxs():
    dset_path = os.path.join(
        dset_dir, '1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-dset.npz'
    )
    model_path = os.path.join(
        './tests/data/models', '1h2o-model.npz'
    )
    dset = dataSet(dset_path)
    model = dict(np.load(model_path, allow_pickle=True))

    n_R = dset.n_R
    n_train = len(model['idxs_train'])
    n_valid = len(model['idxs_valid'])
    n_test = n_R - n_train - n_valid
    
    test_idxs = get_test_idxs(model, dset.asdict(), n_test=None)
    
    assert len(test_idxs) == n_test
