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

# Must be run from mbGDML root directory.

def test_dataset_from_partitioncalc():
    ref_output_path = './tests/data/partition-calcs/out-4H2O-300K-1-ABC.out'
    test_partition = data.PartitionOutput(
        ref_output_path,
        '4H2O',
        'ABC',
        300,
        'hartree',
        'bohr',
        md_iter=1,
        theory='mp2.def2tzvp'
    )
    test_dataset = data.dataSet()
    test_dataset.name_partition(
        test_partition.cluster_label,
        test_partition.partition_label,
        test_partition.md_temp,
        test_partition.md_iter
    )
    assert test_dataset.name == '4H2O-ABC-300K-1-dataset'
    test_dataset.from_partitioncalc(test_partition)

    assert test_dataset.dataset['r_unit'] == 'Angstrom'
    assert np.allclose(test_dataset.dataset['R'][0][4],
                       np.array([-1.39912, -2.017925, -0.902479]))
    assert test_dataset.dataset['e_unit'] == 'kcal/mol'
    assert isclose(test_dataset.dataset['E'][0], -143672.8876798964)
    assert isclose(test_dataset.dataset['E_max'], -143668.556633975)
    assert isclose(test_dataset.dataset['E_mean'], -143671.3661650087)
    assert isclose(test_dataset.dataset['E_min'], -143674.0760573396)
    assert isclose(test_dataset.dataset['E_var'], 1.1866207743)
    assert isclose(test_dataset.dataset['F_max'], 75.0698836347)
    assert np.allclose(test_dataset.dataset['F'][30][0],
                       np.array([6.81548275, -14.34210238, -63.49876045]))
    assert isclose(
        test_dataset.dataset['F_mean'], -2.1959649048e-08, abs_tol=0.0
    )
    assert isclose(test_dataset.dataset['F_min'], -75.0499618465)
    assert isclose(test_dataset.dataset['F_var'], 373.3360402970)
    assert test_dataset.dataset['e_unit'] == 'kcal/mol'
    assert test_dataset.dataset['name'] == '4H2O-ABC-300K-1-dataset'
    assert test_dataset.dataset['system'] == 'solvent'
    assert test_dataset.dataset['solvent'] == 'water'
    assert test_dataset.dataset['cluster_size'] == 3
    assert test_dataset.dataset['theory'] == 'mp2.def2tzvp'


def test_dataset_from_xyz():
    # Energies are hartree
    # Forces are hartree/angstrom
    coord_paths = './tests/data/md/4h2o.abc0-orca.md-mp2.def2tzvp.300k-1.traj'
    force_paths = './tests/data/md/4h2o.abc0-orca.md-mp2.def2tzvp.300k-1.forces'

    test_dataset = data.dataSet()
    test_dataset.read_xyz(
        coord_paths, 'coords', r_unit='Angstrom', energy_comments=True,
        e_unit='hartree'
    )
    test_dataset.read_xyz(force_paths, 'forces')
    assert np.allclose(
        test_dataset.R[13],
        np.array(
            [[ 1.60356892,  1.15940522,  0.45120149],
             [ 2.44897814,  0.98409976, -0.02443958],
             [ 0.91232083,  1.31043934, -0.32129452],
             [-1.38698267, -1.18108194, -0.36717795],
             [-1.24963245, -1.95197596, -0.92292253],
             [-0.81853711, -1.22776338,  0.45900949],
             [-0.43544764,  1.15521674, -1.45105255],
             [-0.8170486 ,  0.26677825, -1.2961774 ],
             [-1.15593236,  1.75261822, -1.2450081 ],
             [ 0.48310408, -1.085288  ,  1.49429163],
             [ 0.25663241, -0.85432318,  2.40218465],
             [ 0.75853895, -0.23317651,  1.09461031]]
        )
    )
    assert test_dataset.E.ndim == 1
    assert test_dataset.E.shape[0] == 101
    assert test_dataset.E[13] == -305.29216392
    assert test_dataset.E_var == 6.919535397308176 * 10**(-6)
    assert test_dataset.F.ndim == 3
    assert np.allclose(
        test_dataset.F[13],
        np.array(
            [[-0.02987315,  0.00745104, -0.08490761],
             [-0.03294625,  0.00602651,  0.02012492],
             [ 0.05300782, -0.01129793,  0.0567368 ],
             [-0.00294407,  0.0179059 ,  0.00730871],
             [ 0.00667818, -0.00859914,  0.01310209],
             [-0.00719587, -0.01668633, -0.03287195],
             [ 0.01113859,  0.00097779, -0.00269464],
             [-0.00129241, -0.00433618,  0.01145648],
             [-0.00686089,  0.00782671, -0.00039129],
             [-0.01617564,  0.00134325,  0.00653721],
             [ 0.00265876, -0.00343129,  0.0014132 ],
             [ 0.02380493,  0.00281966,  0.00418609]]
        )
    )


def test_dataset_from_combined():
    monomer_paths = [
        './tests/data/datasets/A-4H2O-300K-1-dataset.npz',
        './tests/data/datasets/B-4H2O-300K-1-dataset.npz',
        './tests/data/datasets/C-4H2O-300K-1-dataset.npz',
        './tests/data/datasets/D-4H2O-300K-1-dataset.npz'
    ]
    datasetA = data.dataSet(monomer_paths[0])
    datasetB = data.dataSet(monomer_paths[1])
    datasetC = data.dataSet(monomer_paths[2])
    datasetD = data.dataSet(monomer_paths[3])
    
    combined_R = np.concatenate(
        (datasetA.R, datasetB.R, datasetC.R, datasetD.R)
    )
    combined_E = np.concatenate(
        (datasetA.E, datasetB.E, datasetC.E, datasetD.E)
    )
    combined_F = np.concatenate(
        (datasetA.F, datasetB.F, datasetC.F, datasetD.F)
    )

    combined_dataset = data.dataSet()
    combined_dataset.from_combined(monomer_paths)

    assert combined_dataset.type == 'd'
    assert np.allclose(combined_R, combined_dataset.R)
    assert np.allclose(combined_E, combined_dataset.E)
    assert np.allclose(combined_F, combined_dataset.F)
    assert combined_dataset.name == 'A-4H2O-300K-1-dataset'
    combined_dataset.name = '4h2o-monomers-dataset'
    assert combined_dataset.name == '4h2o-monomers-dataset'


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