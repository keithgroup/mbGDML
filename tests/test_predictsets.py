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
from mbgdml.data.predictset import predictSet
import mbgdml.data as data
from mbgdml.mbe import mbePredict
from mbgdml.predict import gdmlModel, predict_gdml, predict_gdml_decomp
from mbgdml.criteria import cm_distance_sum

dset_dir = './tests/data/datasets'
model_dir = './tests/data/models'
molecule_sizes = {
    'h2o': 3,
    'mecn': 6,
    'meoh': 6
}

def test_predictset_correct_contribution_predictions():
    """
    """
    dset_6h2o_path = f'{dset_dir}/6h2o/6h2o.temelso.etal-dset.npz'
    model_h2o_paths = [
        f'{model_dir}/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-model-train500.npz',
        f'{model_dir}/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.2h2o.cm.6-model.mb-train500.npz',
        f'{model_dir}/140h2o.sphere.gfn2.md.500k.prod1.3h2o-model.mb-train500.npz',
    ]
    models = (
        dict(np.load(model_path, allow_pickle=True)) for model_path in model_h2o_paths
    )
    models = [
        gdmlModel(
            model, criteria_desc_func=cm_distance_sum,
            criteria_cutoff=model['cutoff']
        ) for model in models
    ]
    pset = data.predictSet()
    pset.load_dataset(dset_6h2o_path)
    pset.load_models(
        models, predict_gdml_decomp, use_ray=False
    )
    pset.prepare()
    E_pset, F_pset = pset.nbody_predictions([1, 2, 3])

    dset_6h2o = data.dataSet(dset_6h2o_path)
    mbe_pred = mbePredict(models, predict_gdml, use_ray=False)
    E_predict, F_predict = mbe_pred.predict(
        dset_6h2o.z, dset_6h2o.R, dset_6h2o.entity_ids, dset_6h2o.comp_ids,
        ignore_criteria=False
    )
    assert np.allclose(E_pset, E_predict)
    assert np.allclose(F_pset, F_predict)

def test_predictset_nan_for_failed_criteria():
    """Checks that energies and forces are NaN for 
    """
    dset_16h2o_2h2o_path = f'{dset_dir}/2h2o/16h2o.yoo.etal.boat.b.2h2o-dset.mb.npz'
    model_h2o_paths = [
        f'{model_dir}/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.2h2o.cm.6-model.mb-train500.npz',
    ]
    models = (
        dict(np.load(model_path, allow_pickle=True)) for model_path in model_h2o_paths
    )
    models = [
        gdmlModel(
            model, criteria_desc_func=cm_distance_sum,
            criteria_cutoff=model['cutoff']
        ) for model in models
    ]
    pset = data.predictSet()
    pset.load_dataset(dset_16h2o_2h2o_path)
    pset.load_models(models, predict_gdml_decomp)
    pset.prepare()
    E_pset, F_pset = pset.nbody_predictions([2])
    r_isnan = [
        15, 16, 21, 25, 26, 31, 33, 35, 36, 40, 43, 45, 48, 52, 65, 67, 71, 72,
        77, 78, 82, 88, 89, 92, 93, 97, 102, 107, 115, 117
    ]
    for i in range(len(E_pset)):
        if i in r_isnan:
            assert E_pset[i] == 0.0
        else:
            assert E_pset[i] != 0.0
