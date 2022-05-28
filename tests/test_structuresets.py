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

"""Tests for `mbgdml.data.structureset`."""

from math import isclose
import pytest
import numpy as np

from mbgdml.data import structureSet

# Must be run from mbGDML root directory.

def example_10h2o(structureset):
    """

    Parameters
    ----------
    structureset : :obj:`mbgdml.data.structureSet`
    """
    # Information in structure set.
    keys = [
        'type', 'mbgdml_version', 'name', 'R', 'r_unit', 'entity_ids',
        'md5', 'z', 'comp_ids'
    ]
    keys.sort()
    rset_keys = list(structureset.asdict.keys())
    rset_keys.sort()
    assert rset_keys == keys

    # Data type
    assert structureset.type == 's'

    # Name
    assert structureset.name == '10h2o.abc0.iter1.gfn2.md.gfn2.300k.iter1-mbgdml.structset'

    # Atomic numbers are parsed correctly.
    z = np.array([
        8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1,
        8, 1, 1, 8, 1, 1
    ])
    assert np.array_equal(structureset.z, z)
    
    # Has entity_ids.
    entity_ids = np.array([
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
        8, 8, 8, 9, 9, 9
    ])
    assert np.array_equal(structureset.entity_ids, entity_ids)

    # Has comp_ids
    comp_ids = np.array(
        ['h2o', 'h2o', 'h2o', 'h2o', 'h2o', 'h2o', 'h2o', 'h2o', 'h2o', 'h2o']
    )
    assert np.array_equal(structureset.comp_ids, comp_ids)
    
    # Cartesian coordinates
    assert structureset.R.shape == (1000, 30, 3)
    assert structureset.r_unit == 'Angstrom'

    R_32 = np.array(
      [[ 0.2138409 , -0.75317578, -2.48212431],
       [ 0.38055613, -1.52595188, -2.03658303],
       [-0.60671654, -0.21418841, -2.2235319 ],
       [ 1.3443991 ,  1.06432415,  2.17585386],
       [ 1.40262807,  1.36107255,  3.07521589],
       [ 2.23704987,  0.68888485,  1.89998245],
       [-2.37423348,  1.70412911,  0.41693788],
       [-1.5484107 ,  2.2369671 ,  0.44993361],
       [-3.01823576,  2.06705508, -0.12824116],
       [-2.10489634, -0.48412267, -1.35111349],
       [-1.85391629, -1.37302174, -1.16407834],
       [-1.94728712,  0.01046818, -0.58781186],
       [ 0.16968427,  2.6591918 ,  0.10431118],
       [ 0.42945943,  1.8467232 ,  0.5897933 ],
       [ 0.80306083,  2.68773632, -0.57365869],
       [ 2.54191418, -0.29254   ,  0.13050749],
       [ 1.84102152, -0.79450526, -0.32687938],
       [ 2.79765557,  0.35865574, -0.46586859],
       [-1.27296288, -2.45416689,  0.70374188],
       [-1.37855492, -3.28124341,  1.21104697],
       [-1.34421001, -1.8217866 ,  1.52231902],
       [-1.2790591 ,  0.02730456,  2.19232099],
       [-0.45834748,  0.39290178,  2.04539741],
       [-1.91630844,  0.46989582,  1.72914227],
       [ 0.99620097, -2.51709331, -0.32970352],
       [ 0.28924176, -2.13394154,  0.13745409],
       [ 0.96327268, -3.43010222, -0.15911778],
       [ 1.63534332,  1.28281417, -1.86431889],
       [ 1.13350646,  0.40488563, -1.92332564],
       [ 1.96419952,  1.76601828, -2.58558329]]
    )
    assert np.allclose(R_32, structureset.R[32])

    # MD5 hash
    assert structureset.md5 == 'cd0cfdc29ebc52eb40d2a028124cfd18'

def example_6h2o_md(structureset):
    """

    Parameters
    ----------
    structureset : :obj:`mbgdml.data.structureSet`
    """
    # Information in structure set.
    keys = [
        'type', 'mbgdml_version', 'name', 'R', 'r_unit', 'entity_ids',
        'md5', 'z', 'comp_ids'
    ]
    keys.sort()
    rset_keys = list(structureset.asdict.keys())
    rset_keys.sort()
    assert rset_keys == keys

    
    # Data type
    assert structureset.type == 's'
    
    # Name
    assert structureset.name == '6h2o.temelso.etal.pr.md.gfn2.300k.step10000-ase.md-orca.mp2.def2tzvp.300k'
    
    # Atomic numbers are parsed correctly.
    z = np.array([
        8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1
    ])
    assert np.array_equal(structureset.z, z)
    
    # Has entity_ids.
    entity_ids = np.array([
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5
    ])
    assert np.array_equal(structureset.entity_ids, entity_ids)

    # Has comp_ids
    comp_ids = np.array(['h2o', 'h2o', 'h2o', 'h2o', 'h2o', 'h2o'])
    assert np.array_equal(structureset.comp_ids, comp_ids)
    
    # Cartesian coordinates
    assert structureset.R.shape == (301, 18, 3)
    assert structureset.r_unit == 'Angstrom'

    R_32 = np.array(
      [[0.2027597,-4.00281929,-0.90995974],
        [-0.16286238,-3.14967717,-0.67903394],
        [0.48625758,-4.48705124,-0.09796301],
        [-1.95833273,-0.74515469,-1.21757579],
        [-1.99166926,-0.43717089,-2.1486605,],
        [-2.12546867,0.05197179,-0.64383929],
        [-2.14044116,1.60742354,0.38325509],
        [-1.37023851,2.04184207,0.61056076],
        [-2.85820184,2.17750563,0.63762777],
        [2.80819533,0.63151693,1.62568434],
        [2.19506056,-0.15937379,1.50464908],
        [3.2118317,0.39423199,2.43503994],
        [0.79762923,-0.51292892,-0.20285553],
        [-0.14485871,-0.38287672,-0.56317643],
        [1.35827021,-0.6160705,-1.00482268],
        [0.38018651,2.56275112,0.53046884],
        [1.18552353,3.09643848,0.63682096],
        [0.81705268,1.68340128,0.54155129]]
    )
    assert np.allclose(R_32, structureset.R[32])
    
    # MD5 hash
    assert structureset.md5 == 'dd6d875868a04e3c3a5deb71e280371c'

def test_structureset_from_traj():
    traj_path = './tests/data/md/10h2o.abc0.iter1.gfn2-xtb.md-gfn2.300k-1.traj'

    # Getting entity_ids.
    h2o_size = 3
    cluster_size = 10
    entity_ids = []
    for i in range(0, cluster_size):
        entity_ids.extend([i for _ in range(0, h2o_size)])

    # Getting comp_ids
    solvent = 'h2o'
    comp_ids = []
    for i in range(0, cluster_size):
        comp_ids.append(solvent)

    # Creating structure set.
    test_structureset = structureSet()
    test_structureset.from_xyz(traj_path, 'Angstrom', entity_ids, comp_ids)

    # Naming of the structure set.
    assert test_structureset.name == '10h2o.abc0.iter1.gfn2-xtb.md-gfn2.300k-1'
    test_structureset.name = '10h2o.abc0.iter1.gfn2.md.gfn2.300k.iter1-mbgdml.structset'

    # test_structureset.save(test_structureset.name, test_structureset.structureset, './tests/data/structuresets/')

    example_10h2o(test_structureset)

def test_structureset_load():
    structureset_path = './tests/data/structuresets/10h2o.abc0.iter1.gfn2.md.gfn2.300k.iter1-mbgdml.structset.npz'

    test_structureset = structureSet(structureset_path)

    example_10h2o(test_structureset)

def test_structureset_from_npz():
    npz_path = './tests/data/md/6h2o.temelso.etal.pr.md.gfn2.300k.step10000-ase.md-orca.mp2.def2tzvp.300k.npz'

    # Getting entity_ids.
    h2o_size = 3
    cluster_size = 6
    entity_ids = []
    for i in range(0, cluster_size):
        entity_ids.extend([i for _ in range(0, h2o_size)])

    # Getting comp_ids
    solvent = 'h2o'
    comp_ids = []
    for i in range(0, cluster_size):
        comp_ids.append(solvent)

    # Creating structure set.
    test_structureset = structureSet()
    test_structureset.from_npz(npz_path, 'z', 'R', 'Angstrom', entity_ids, comp_ids)

    # Naming of the structure set.
    assert test_structureset.name == '6h2o.temelso.etal.pr.md.gfn2.300k.step10000-ase.md-orca.mp2.def2tzvp.300k'
    test_structureset.name = '6h2o.temelso.etal.pr.md.gfn2.300k.step10000-ase.md-orca.mp2.def2tzvp.300k'

    example_6h2o_md(test_structureset)
    