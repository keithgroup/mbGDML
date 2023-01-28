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
from mbgdml.predictors import predict_gdml
from mbgdml.descriptors import Criteria, com_distance_sum

dset_dir = "./tests/data/datasets"
model_dir = "./tests/data/models"
molecule_sizes = {"h2o": 3, "mecn": 6, "meoh": 6}


def test_predict_single_16mer():
    dset_16h2o_path = f"{dset_dir}/16h2o/16h2o.yoo.etal.boat.b-dset-mp2.def2tzvp.npz"
    model_h2o_paths = [
        "140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-model-train500.npz",
        "140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.2h2o.cm.6-model.mb-train500.npz",
        "140h2o.sphere.gfn2.md.500k.prod1.3h2o-model.mb-train500.npz",
    ]
    model_h2o_paths = [f"{model_dir}/{path}" for path in model_h2o_paths]
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

    dset_16h2o = data.DataSet(dset_16h2o_path, Z_key="z")
    mbe_pred = mbePredict(models, predict_gdml, use_ray=False)
    E_predict, F_predict = mbe_pred.predict(
        dset_16h2o.Z, dset_16h2o.R, dset_16h2o.entity_ids, dset_16h2o.comp_ids
    )
    E = np.array([-766368.03399751])
    F = np.array(
        [
            [
                [0.29906572, 0.14785963, 0.24781407],
                [-0.30412644, -0.72411633, -0.11358761],
                [-0.49192677, 0.86896897, -0.67525678],
                [0.36627638, 1.02869105, -2.56223656],
                [-0.10503164, -0.89234795, 0.9294424],
                [-0.1841222, -0.14389019, 1.2193703],
                [-1.38995634, 1.74512784, 0.20352509],
                [0.50352734, -1.84912139, -1.11214437],
                [-0.45073645, -0.58830104, -0.0708215],
                [-0.05824096, -0.07168296, 3.05363522],
                [-0.21573588, 0.55601679, -0.93232724],
                [0.33556773, 0.3464968, -1.20999654],
                [1.13396357, 0.64719014, -0.37314183],
                [-0.14864126, -0.74782087, 0.92789942],
                [0.25446292, 0.18875155, 0.35677525],
                [1.18808078, 0.9989521, -1.70936528],
                [-0.42772192, -0.23482216, 2.22942188],
                [0.5023115, -0.2546999, 0.59431561],
                [1.03039212, -0.27777061, 0.43893643],
                [-1.6481248, -0.11736926, 0.39427926],
                [-0.8270073, -1.08703941, -0.46220551],
                [-1.65290086, -0.85447434, -0.25093955],
                [2.38457849, -0.51709509, -0.97800052],
                [0.70822521, 0.11395345, 1.4606325],
                [-0.49915379, 2.60146319, 1.20100891],
                [-0.01957611, -1.61507913, -0.3507438],
                [-0.04340775, -0.95576235, -0.88557194],
                [-0.1068999, -1.47361438, -0.57488098],
                [0.10196448, 1.2622373, -0.57288566],
                [0.46155007, 0.86992573, -0.07612512],
                [-0.06659418, -1.53956909, -2.77945064],
                [-0.30081568, 0.14797997, 0.90844867],
                [0.38111199, 1.29149786, 0.63063523],
                [0.27202453, 0.04869613, -1.44668878],
                [0.03618388, -0.62330206, -1.39043361],
                [-0.5954522, 0.61790128, 1.67910304],
                [0.10622445, 0.31818432, 0.72714358],
                [-0.48496294, 0.85814888, -0.29055761],
                [-0.85844605, 0.18657187, -0.07795668],
                [2.58353778, -0.54173036, 0.4635027],
                [-1.56162087, 0.12760808, 0.02244887],
                [-0.65542649, 0.34366634, 0.19180049],
                [-2.35675996, -1.09049215, 0.22829278],
                [0.71868199, 0.072091, -0.36158273],
                [1.55157057, 0.37661812, -0.25918432],
                [-1.39910186, -0.24662851, 2.7263307],
                [1.55454091, 0.60506067, -1.08736517],
                [0.3786482, 0.07707048, -0.23131207],
            ]
        ]
    )

    assert np.allclose(E_predict, E)
    assert np.allclose(F_predict, F, rtol=1e-04, atol=1e-02)
