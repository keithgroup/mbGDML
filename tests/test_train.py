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

"""Tests for GDML training"""

# pylint: skip-file

import os
import numpy as np
from mbgdml.data import DataSet
from mbgdml._gdml.train import GDMLTrain, get_test_idxs
from mbgdml.train import mbGDMLTrain
from mbgdml.analysis.problematic import ProblematicStructures
from mbgdml.models import gdmlModel
from mbgdml.predictors import predict_gdml
from mbgdml.descriptors import Criteria, com_distance_sum

DSET_DIR = "./tests/data/datasets"
TRAIN_DIR = "./tests/data/train"


def check_glob():
    # pylint: disable=undefined-variable, global-variable-undefined
    global glob
    if "glob" in globals():
        del glob


def test_train_results_1h2o():
    r"""Checks the results of a training task."""
    check_glob()

    dset_path = os.path.join(
        DSET_DIR, "1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-dset.npz"
    )
    dset = DataSet(dset_path, Z_key="z")
    dset_dict = dset.asdict(gdml_keys=True)

    train_dir_1h2o = os.path.join(TRAIN_DIR, "1h2o/")
    train_idxs_path = os.path.join(train_dir_1h2o, "train_idxs.npy")
    valid_idxs_path = os.path.join(train_dir_1h2o, "valid_idxs.npy")
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
        solver="analytic",
        solver_tol=1e-4,
        idxs_train=train_idxs,
        idxs_valid=valid_idxs,
    )
    model = train.train(task)

    # Reference data
    # pylint: disable-next=invalid-name
    alphas_F_ref = np.load(
        os.path.join(train_dir_1h2o, "alphas_F.npy"), allow_pickle=True
    )
    # pylint: disable-next=invalid-name
    R_desc_ref = np.load(os.path.join(train_dir_1h2o, "R_desc.npy"), allow_pickle=True)
    tril_perms_lin_ref = np.load(
        os.path.join(train_dir_1h2o, "tril_perms_lin.npy"), allow_pickle=True
    )

    del train

    # Coefficients will not be exactly the same.
    assert np.allclose(model["R_desc"], R_desc_ref, rtol=1e-05, atol=1e-08)
    assert np.allclose(model["alphas_F"], alphas_F_ref, rtol=1e2, atol=0)
    assert np.allclose(np.array(model["c"]), np.array(331288.48632617114))
    assert np.allclose(
        np.array(model["norm_y_train"]), np.array(321987215081.7051), rtol=1e-3, atol=0
    )
    assert np.allclose(
        np.array(model["tril_perms_lin"]), tril_perms_lin_ref, rtol=1e-3, atol=0
    )


def test_1h2o_train_grid_search():
    check_glob()

    dset_path = os.path.join(
        DSET_DIR, "1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-dset.npz"
    )
    dset = DataSet(dset_path, Z_key="z")

    train_dir_1h2o = os.path.join(TRAIN_DIR, "1h2o/")
    train_idxs_path = os.path.join(train_dir_1h2o, "train_idxs.npy")
    valid_idxs_path = os.path.join(train_dir_1h2o, "valid_idxs.npy")
    train_idxs = np.load(train_idxs_path, allow_pickle=True)
    valid_idxs = np.load(valid_idxs_path, allow_pickle=True)

    n_train = 50
    n_valid = 100
    sigma_grid = [32, 42, 52]

    train = mbGDMLTrain(
        entity_ids=np.array([0, 0, 0]),
        comp_ids=np.array(["h2o"]),
        use_sym=True,
        use_E=True,
        use_E_cstr=False,
        use_cprsn=False,
        solver="analytic",
        lam=1e-15,
        solver_tol=1e-4,
    )
    train.sigma_grid = sigma_grid
    model = train.grid_search(
        dset,
        "1h2o",
        n_train,
        n_valid,
        train_idxs=train_idxs,
        valid_idxs=valid_idxs,
        write_json=True,
        write_idxs=True,
        overwrite=True,
        save_dir="./tests/tmp/1h2o-grid",
    )

    del train

    assert model["sig"].item() == 42
    assert np.allclose(
        np.array(model["f_err"].item()["rmse"]),
        0.4673519840841512,
        rtol=1e-05,
        atol=1e-08,
    )
    assert model["perms"].shape[0] == 2


def test_1h2o_train_grid_search_iterative():
    check_glob()

    dset_path = os.path.join(
        DSET_DIR, "1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-dset.npz"
    )
    dset = DataSet(dset_path, Z_key="z")

    train_dir_1h2o = os.path.join(TRAIN_DIR, "1h2o/")
    train_idxs_path = os.path.join(train_dir_1h2o, "train_idxs.npy")
    valid_idxs_path = os.path.join(train_dir_1h2o, "valid_idxs.npy")
    train_idxs = np.load(train_idxs_path, allow_pickle=True)
    valid_idxs = np.load(valid_idxs_path, allow_pickle=True)

    n_train = 50
    n_valid = 100
    sigma_grid = [32, 42, 52]

    train = mbGDMLTrain(
        entity_ids=np.array([0, 0, 0]),
        comp_ids=np.array(["h2o"]),
        use_sym=True,
        use_E=True,
        use_E_cstr=False,
        use_cprsn=False,
        solver="iterative",
        lam=1e-15,
        solver_tol=1e-4,
    )
    train.sigma_grid = sigma_grid
    model = train.grid_search(
        dset,
        "1h2o",
        n_train,
        n_valid,
        train_idxs=train_idxs,
        valid_idxs=valid_idxs,
        write_json=True,
        write_idxs=True,
        overwrite=True,
        save_dir="./tests/tmp/1h2o-grid-iterative",
    )

    del train

    assert model["sig"].item() == 42
    assert np.allclose(
        np.array(model["f_err"].item()["rmse"]),
        0.4673519840841512,
        rtol=1e-02,
        atol=1e-03,
    )
    assert model["perms"].shape[0] == 2


def test_1h2o_train_bayes_opt():
    check_glob()

    dset_path = os.path.join(
        DSET_DIR, "1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-dset.npz"
    )
    dset = DataSet(dset_path, Z_key="z")

    train_dir_1h2o = os.path.join(TRAIN_DIR, "1h2o/")
    train_idxs_path = os.path.join(train_dir_1h2o, "train_idxs.npy")
    valid_idxs_path = os.path.join(train_dir_1h2o, "valid_idxs.npy")
    train_idxs = np.load(train_idxs_path, allow_pickle=True)
    valid_idxs = np.load(valid_idxs_path, allow_pickle=True)

    n_train = 50
    n_valid = 100
    sigma_grid = [32, 42, 52]

    train = mbGDMLTrain(
        entity_ids=np.array([0, 0, 0]),
        comp_ids=np.array(["h2o"]),
        use_sym=True,
        use_E=True,
        use_E_cstr=False,
        use_cprsn=False,
        solver="analytic",
        lam=1e-15,
        solver_tol=1e-4,
    )
    train.sigma_grid = sigma_grid
    train.bayes_opt_params = {"init_points": 5, "n_iter": 5, "alpha": 1e-7}
    train.sigma_bounds = (2, 100)
    model, _ = train.bayes_opt(
        dset,
        "1h2o",
        n_train,
        n_valid,
        save_dir="./tests/tmp/1h2o-bo",
        train_idxs=train_idxs,
        valid_idxs=valid_idxs,
        overwrite=True,
        write_json=True,
        write_idxs=True,
    )

    assert model["perms"].shape[0] == 2

    assert model["f_err"].item()["rmse"] < 0.469
    assert model["e_err"].item()["rmse"] < 0.0251

    del train


def test_1h2o_prob_indices():
    check_glob()

    dset_path = os.path.join(
        DSET_DIR, "1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-dset.npz"
    )
    model_path = os.path.join("./tests/data/models", "1h2o-model.npz")
    model_dict = dict(np.load(model_path, allow_pickle=True))
    model_desc_kwargs = {"entity_ids": np.array([0, 0, 0])}
    model_desc_cutoff = None
    model_criteria = Criteria(com_distance_sum, model_desc_kwargs, model_desc_cutoff)
    model = gdmlModel(model_dict, criteria=model_criteria)
    dset = DataSet(dset_path, Z_key="z")

    prob_s = ProblematicStructures([model], predict_gdml)
    n_find = 100
    prob_idxs = prob_s.find(dset, n_find, save_dir="./tests/tmp")
    prob_idxs = np.sort(prob_idxs)

    ref = np.array(
        [
            465,
            541,
            653,
            798,
            807,
            921,
            953,
            1058,
            1240,
            1421,
            1430,
            1510,
            1618,
            1663,
            1665,
            1676,
            1890,
            2090,
            2123,
            2218,
            2246,
            2665,
            2944,
            3171,
            3225,
            3485,
            3510,
            3738,
            3795,
            3970,
            3994,
            4272,
            4660,
            5102,
            5150,
            5195,
            5230,
            6394,
            6471,
            6787,
            6900,
            6961,
            6986,
            7257,
            7725,
            7735,
            7812,
            7815,
            8006,
            8074,
            8253,
            8489,
            8532,
            8810,
            9169,
            9221,
            9226,
            9667,
            9668,
            9728,
            9747,
            9919,
            9952,
            9995,
            10025,
            10057,
            10062,
            10144,
            10252,
            10525,
            10763,
            10982,
            11005,
            11012,
            11024,
            11404,
            11730,
            11745,
            11747,
            11864,
            11970,
            12049,
            12167,
            12329,
            12465,
            12478,
            12638,
            12645,
            12655,
            12664,
            12775,
            12878,
            13062,
            13151,
            13192,
            13320,
            13343,
            13546,
            13676,
            13963,
        ]
    )

    assert len(prob_idxs) == 100
    # This is a very bad test, but will work for now?
    assert len(np.setdiff1d(prob_idxs, ref)) < 20


def test_getting_test_idxs():
    dset_path = os.path.join(
        DSET_DIR, "1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.dset.1h2o-dset.npz"
    )
    model_path = os.path.join("./tests/data/models", "1h2o-model.npz")
    dset = DataSet(dset_path, Z_key="z")
    model = dict(np.load(model_path, allow_pickle=True))

    n_R = dset.n_R
    n_train = len(model["idxs_train"])
    n_valid = len(model["idxs_valid"])
    n_test = n_R - n_train - n_valid

    test_idxs = get_test_idxs(model, dset.asdict(), n_test=None)

    assert len(test_idxs) == n_test
