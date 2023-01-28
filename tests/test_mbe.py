# MIT License
#
# Copyright (c) 2022-2023, Alex M. Maldonado
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

"""Tests many-body utilities"""

# pylint: skip-file

import numpy as np
from mbgdml.mbe import mbe_contrib

h2o_4mer_data = {
    "4h2o": dict(np.load("./tests/data/mbe/4h2o.npz", allow_pickle=True)),
    "1body": dict(np.load("./tests/data/mbe/4h2o-1body.npz", allow_pickle=True)),
    "2body": dict(np.load("./tests/data/mbe/4h2o-2body.npz", allow_pickle=True)),
    "3body": dict(np.load("./tests/data/mbe/4h2o-3body.npz", allow_pickle=True)),
}


def test_4h2o_mbe_from_data():
    r"""Many-body prediction of water tetramers."""

    E_true = h2o_4mer_data["4h2o"]["energy_ele_mp2.def2tzvp_orca"]
    G_true = h2o_4mer_data["4h2o"]["grads_mp2.def2tzvp_orca"]
    assert np.allclose(E_true, np.array([-305.30169218, -305.2999005, -305.29520327]))
    E_mb = np.zeros(E_true.shape)
    G_mb = np.zeros(G_true.shape)
    entity_ids = h2o_4mer_data["4h2o"]["entity_ids"]
    r_prov_ids = None
    r_prov_specs = None

    def get_lower_data(data, e_key, g_key):
        E_lower = data[e_key]
        G_lower = data[g_key]
        entity_ids_lower = data["entity_ids"]
        r_prov_ids_lower = data["r_prov_ids"].item()
        r_prov_specs_lower = data["r_prov_specs"]
        return E_lower, G_lower, entity_ids_lower, r_prov_ids_lower, r_prov_specs_lower

    lower_data = []
    lower_data.append(
        (
            get_lower_data(
                h2o_4mer_data["1body"],
                "energy_ele_mp2.def2tzvp_orca",
                "grads_mp2.def2tzvp_orca",
            )
        )
    )
    lower_data.append(
        (
            get_lower_data(
                h2o_4mer_data["2body"],
                "energy_ele_nbody_mp2.def2tzvp_orca",
                "grads_nbody_mp2.def2tzvp_orca",
            )
        )
    )
    lower_data.append(
        (
            get_lower_data(
                h2o_4mer_data["3body"],
                "energy_ele_nbody_mp2.def2tzvp_orca",
                "grads_nbody_mp2.def2tzvp_orca",
            )
        )
    )

    for data in lower_data:
        E_mb, G_mb = mbe_contrib(
            E_mb,
            G_mb,
            entity_ids,
            r_prov_ids,
            r_prov_specs,
            *data,
            operation="add",
            use_ray=False
        )

    E_ref = np.array([-305.30075391, -305.29900387, -305.29489548])
    assert np.allclose(E_mb, E_ref)
