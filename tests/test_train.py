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
from mbgdml.gdml.train import GDMLTrain

dset_dir = './tests/data/datasets'
train_dir = './tests/data/train'

def test_train_results_1h2o():
    """Checks the results of a training task."""
    dset_path = os.path.join(
        dset_dir, '1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-dset.npz'
    )
    dset = dataSet(dset_path)
    dset_dict = dset.asdict

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
        n_inducing_pts_init=25,
        interact_cut_off=None,
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

    assert np.allclose(alphas_F, alphas_F_ref)
    assert np.allclose(R_desc, R_desc_ref)
    assert model['c'] == 331288.48632617114
    assert model['norm_y_train'] == 321987215081.7051
