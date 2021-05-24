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
import mbgdml.parse as parse
import mbgdml.utils as utils
import mbgdml.calculate as calculate

# Must be run from mbGDML root directory.
    
def test_calculate_ORCA():
    z_test = np.array([8, 1, 1])
    R_test = np.array(
        [[-0.48381516,  1.17384211, -1.4413092 ],
         [-0.90248552,  0.33071306, -1.24479905],
         [-1.21198585,  1.83409853, -1.4187445 ]]
    )
    coord_string = utils.string_coords(z_test, R_test)
    assert coord_string == ("O  -0.483815160   1.173842110  -1.441309200\n"
                           "H  -0.902485520   0.330713060  -1.244799050\n"
                           "H  -1.211985850   1.834098530  -1.418744500\n")
    
    engrad = calculate.ORCA(
        '4h2o.abc0.2.step2',
        '4h2o.abc0.2.step2-orca.engrad-mp2.def2tzvp.tightscf.frozencore',
        '4h2o.abc0.2.step2-orca.engrad-mp2.def2tzvp.tightscf.frozencore'
    )
    slurm_file_name, slurm_file = engrad.submit(
        'smd',
        1,
        6,
        0,
        12,
        calculate.pitt_crc_orca_submit,
        write=False,
    )
    
    assert slurm_file_name == 'submit-orca.420.slurm'
    assert slurm_file == (
        "#!/bin/bash\n"
        "#SBATCH --job-name=4h2o.abc0.2.step2\n"
        "#SBATCH --output=4h2o.abc0.2.step2-orca.engrad-mp2.def2tzvp.tightscf.frozencore.out\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --ntasks-per-node=6\n"
        "#SBATCH --time=0-12:00:00\n"
        "#SBATCH --cluster=smd\n"
        "\n"
        "cd $SBATCH_O_WORKDIR\n"
        "module purge\n"
        "module load openmpi/3.1.4\n"
        "module load orca/4.2.0\n"
        "\n"
        "cp $SLURM_SUBMIT_DIR/*.inp $SLURM_SCRATCH\n"
        "\n"
        "export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH\n"
        "# Suppresses OpenFabrics (openib) absence nonfatal error.\n"
        "export OMPI_MCA_btl_base_warn_component_unused=0\n"
        "# Makes all error messages print.\n"
        "export OMPI_MCA_orte_base_help_aggregate=0"
        "\n"
        "\n"
        "cd $SLURM_SCRATCH"
        "\n"
        "$(which orca) *.inp"
        "\n"
        "\n"
    )

    step_R_string = utils.string_coords(z_test, R_test)
    assert step_R_string == (
        "O  -0.483815160   1.173842110  -1.441309200\n"
        "H  -0.902485520   0.330713060  -1.244799050\n"
        "H  -1.211985850   1.834098530  -1.418744500\n"
    )
    _, calc_string = engrad.input(
        'EnGrad',
        step_R_string,
        'MP2',
        'def2-TZVP',
        0,
        1,
        6,
        options='TightSCF FrozenCore',
        control_blocks='%scf\n    ConvForced true\nend\n%maxcore 8000\n',
        write=False
    )
    assert calc_string == (
        "# 4h2o.abc0.2.step2\n"
        "! MP2 def2-TZVP EnGrad TightSCF FrozenCore\n"
        "\n"
        "%pal\n"
        "    nprocs 6\n"
        "end\n"
        "\n"
        "%scf\n"
        "    ConvForced true\n"
        "end\n"
        "%maxcore 8000\n"
        "\n"
        "\n"
        "*xyz 0 1\n"
        "O  -0.483815160   1.173842110  -1.441309200\n"
        "H  -0.902485520   0.330713060  -1.244799050\n"
        "H  -1.211985850   1.834098530  -1.418744500\n"
        "*\n"
    )
