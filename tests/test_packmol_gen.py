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

"""Tests generating packmol structures"""

# pylint: skip-file

import os
import shutil
import pytest
import numpy as np

from mbgdml.structure_gen.packmol_gen import get_packmol_input, run_packmol

data_dir = "./tests/data"


def test_packmol_water_box():
    packmol_path = shutil.which("packmol")
    if packmol_path is None:
        pytest.skip("packmol is not installed")

    work_dir = "./tests/tmp/packmol-tests"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    packmol_input_lines = get_packmol_input(
        "box",
        10.0,
        np.array([33]),
        os.path.join(data_dir, "structures/h2o.xyz"),
        output_path=os.path.join(work_dir, "water-box.xyz"),
        periodic=True,
        dist_tolerance=2.0,
        filetype="xyz",
        seed=20276470,
        periodic_shift=1.0,
    )
    ref_input_lines = [
        "tolerance 2.0\n",
        "seed 20276470\n",
        "filetype xyz\n\n",
        "structure ./tests/data/structures/h2o.xyz\n",
        "    number 33\n",
        "    inside box 1.0 1.0 1.0 9.0 9.0 9.0\n",
        "end structure\n\n",
        "output ./tests/tmp/packmol-tests/water-box.xyz\n",
    ]
    assert packmol_input_lines == ref_input_lines

    Z, R = run_packmol(work_dir, packmol_input_lines, packmol_path="packmol")
    Z_ref = np.array(
        [
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
            8,
            1,
            1,
        ],
        dtype=np.uint8,
    )
    R_ref = np.array(
        [
            [4.448288, 1.208086, 2.746892],
            [3.536319, 0.983195, 2.955685],
            [4.512732, 1.100818, 1.792849],
            [3.948733, 1.316616, 6.793027],
            [3.853295, 2.273996, 6.806119],
            [3.250261, 1.010757, 6.206132],
            [8.520903, 5.980787, 8.300956],
            [8.02853, 5.491196, 8.967086],
            [7.85672, 6.256644, 7.66172],
            [3.049999, 7.894083, 5.432018],
            [3.435076, 7.885144, 6.313774],
            [3.536801, 7.220277, 4.947388],
            [5.809833, 8.75848, 8.13287],
            [5.08962, 9.004536, 7.544135],
            [5.378026, 8.335505, 8.881531],
            [8.959022, 3.79945, 7.522505],
            [8.105949, 4.116766, 7.210358],
            [9.085442, 2.958855, 7.071643],
            [6.721564, 7.502117, 2.659343],
            [7.427039, 7.328975, 3.290362],
            [5.947224, 7.678085, 3.202749],
            [6.39461, 5.178943, 6.110874],
            [5.559335, 5.040733, 5.653631],
            [7.067268, 4.938945, 5.466056],
            [5.896766, 5.537198, 8.219439],
            [5.938502, 6.136762, 8.970864],
            [6.029607, 4.662834, 8.598515],
            [1.024728, 3.675127, 4.249228],
            [1.003345, 4.457946, 4.808331],
            [1.570091, 3.92793, 3.497877],
            [8.331325, 8.604727, 7.490032],
            [8.408019, 8.3271, 8.408128],
            [8.943106, 8.036109, 7.012272],
            [7.645795, 2.445103, 8.62787],
            [8.006798, 1.618302, 8.962437],
            [6.764088, 2.497423, 9.009594],
            [1.01286, 6.70738, 3.90794],
            [1.849672, 6.236197, 3.848027],
            [0.98588, 7.041051, 4.810045],
            [1.084508, 1.394258, 4.855807],
            [1.918032, 1.161958, 4.434934],
            [1.333853, 1.960597, 5.592656],
            [8.159789, 8.976462, 1.691952],
            [7.491096, 8.99982, 1.000459],
            [8.996631, 8.929985, 1.219307],
            [2.855454, 5.441794, 6.042908],
            [2.200437, 6.069425, 6.363674],
            [3.644715, 5.628257, 6.560745],
            [1.114017, 1.632434, 2.395827],
            [1.342919, 1.006683, 1.701638],
            [1.006312, 2.472781, 1.939675],
            [1.011581, 8.889374, 3.472575],
            [1.546404, 8.833852, 2.674616],
            [1.646378, 9.034334, 4.181009],
            [4.188229, 3.228318, 4.365115],
            [3.885257, 3.724723, 5.131696],
            [4.095151, 2.305409, 4.620915],
            [8.647352, 6.535042, 1.015165],
            [9.010228, 6.684375, 1.893731],
            [7.801287, 6.104299, 1.171665],
            [1.020174, 8.071019, 8.119338],
            [1.055278, 8.326201, 7.192241],
            [1.048206, 8.90314, 8.601673],
            [9.018138, 6.527312, 5.336616],
            [8.117733, 6.866508, 5.345254],
            [8.982312, 5.757471, 4.760492],
            [7.337606, 1.034847, 6.955019],
            [8.17008, 1.05466, 6.472884],
            [6.661836, 1.134434, 6.27732],
            [4.179361, 5.717711, 2.552372],
            [3.333954, 6.050423, 2.235436],
            [3.953449, 5.145917, 3.292557],
            [3.645005, 4.397988, 8.472033],
            [3.92885, 5.095361, 9.071167],
            [3.373635, 3.677386, 9.04905],
            [6.277071, 2.534167, 4.639107],
            [7.162709, 2.904082, 4.707426],
            [6.194422, 2.270085, 3.717539],
            [4.668657, 1.829959, 8.805258],
            [5.173153, 1.037956, 8.595323],
            [3.783191, 1.508985, 9.002196],
            [6.031231, 4.470176, 3.585363],
            [6.03153, 4.193076, 2.663911],
            [6.120551, 5.427629, 3.551246],
            [1.586099, 1.466764, 8.924615],
            [0.994265, 2.161784, 8.620414],
            [1.835234, 0.992591, 8.125271],
            [2.995187, 7.565108, 8.947375],
            [2.925579, 6.882351, 8.272945],
            [3.193377, 8.370016, 8.458805],
            [1.610803, 4.059269, 7.70748],
            [1.993036, 3.406308, 7.113007],
            [0.945326, 4.509693, 7.17822],
            [6.288949, 7.378714, 6.467604],
            [6.605633, 8.230985, 6.152638],
            [5.488556, 7.213994, 5.959572],
            [8.406661, 2.835502, 2.118919],
            [8.750038, 3.656909, 2.483939],
            [8.229177, 2.283705, 2.886954],
        ],
        dtype=np.float64,
    )
    assert np.allclose(Z, Z_ref)
    assert np.allclose(R, R_ref, rtol=0.1, atol=0.1)
