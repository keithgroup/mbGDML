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

import numpy as np
from mbgdml import data
from mbgdml.mbe import mbePredict
from mbgdml.models import gdmlModel
from mbgdml.predictors import predict_gdml, predict_gdml_decomp
from mbgdml.descriptors import Criteria, com_distance_sum

DSET_DIR = "./tests/data/datasets"
MODEL_DIR = "./tests/data/models"
molecule_sizes = {"h2o": 3, "mecn": 6, "meoh": 6}


def test_predictset_correct_contribution_predictions():
    # pylint: disable=line-too-long
    dset_6h2o_path = f"{DSET_DIR}/6h2o/6h2o.temelso.etal-dset.npz"
    model_h2o_paths = [
        f"{MODEL_DIR}/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-model-train500.npz",
        f"{MODEL_DIR}/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.2h2o.cm.6-model.mb-train500.npz",
        f"{MODEL_DIR}/140h2o.sphere.gfn2.md.500k.prod1.3h2o-model.mb-train500.npz",
    ]
    model_dicts = (
        dict(np.load(model_path, allow_pickle=True)) for model_path in model_h2o_paths
    )
    model_desc_kwargs = (
        {"entity_ids": np.array([0, 0, 0])},
        {"entity_ids": np.array([0, 0, 0, 1, 1, 1])},
        {"entity_ids": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])},
    )
    model_desc_cutoffs = (None, 6.0, 10.0)
    model_criterias = [
        Criteria(com_distance_sum, desc_kwargs, cutoff)
        for desc_kwargs, cutoff in zip(model_desc_kwargs, model_desc_cutoffs)
    ]
    models = [
        gdmlModel(model, criteria=criteria)
        for model, criteria in zip(model_dicts, model_criterias)
    ]
    pset = data.PredictSet()
    pset.load_dataset(dset_6h2o_path, Z_key="z")
    pset.load_models(models, predict_gdml_decomp, use_ray=False)
    pset.prepare()
    E_pset, F_pset = pset.nbody_predictions([1, 2, 3])

    dset_6h2o = data.DataSet(dset_6h2o_path, Z_key="z")
    mbe_pred = mbePredict(models, predict_gdml, use_ray=False)
    E_predict, F_predict = mbe_pred.predict(
        dset_6h2o.Z, dset_6h2o.R, dset_6h2o.entity_ids, dset_6h2o.comp_ids
    )
    assert np.allclose(E_pset, E_predict)
    assert np.allclose(F_pset, F_predict)


def test_predictset_nan_for_failed_criteria():
    r"""Checks that energies and forces are NaN for"""
    # pylint: disable=line-too-long
    dset_16h2o_2h2o_path = f"{DSET_DIR}/2h2o/16h2o.yoo.etal.boat.b.2h2o-dset.mb.npz"
    model_h2o_paths = [
        f"{MODEL_DIR}/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.2h2o.cm.6-model.mb-train500.npz",
    ]
    model_dicts = (
        dict(np.load(model_path, allow_pickle=True)) for model_path in model_h2o_paths
    )
    model_desc_kwargs = ({"entity_ids": np.array([0, 0, 0, 1, 1, 1])},)
    model_desc_cutoffs = (6.0,)
    model_criterias = [
        Criteria(com_distance_sum, desc_kwargs, cutoff)
        for desc_kwargs, cutoff in zip(model_desc_kwargs, model_desc_cutoffs)
    ]
    models = [
        gdmlModel(model, criteria=criteria)
        for model, criteria in zip(model_dicts, model_criterias)
    ]
    pset = data.PredictSet()
    pset.load_dataset(dset_16h2o_2h2o_path, Z_key="z")
    pset.load_models(models, predict_gdml_decomp)
    pset.prepare()
    E_pset, _ = pset.nbody_predictions([2])
    r_isnan = [
        15,
        16,
        21,
        25,
        26,
        31,
        33,
        35,
        36,
        40,
        43,
        45,
        48,
        52,
        65,
        67,
        71,
        72,
        77,
        78,
        82,
        88,
        89,
        92,
        93,
        97,
        102,
        107,
        115,
        117,
    ]
    # pylint: disable-next=consider-using-enumerate
    for i in range(len(E_pset)):
        if i in r_isnan:
            assert np.isnan(E_pset[i])
        else:
            assert not np.isnan(E_pset[i])
