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

"""Tests generating structure generation utilities"""

# pylint: skip-file

import numpy as np

from mbgdml.structure_gen.utils import get_num_mols

data_dir = "./tests/data"


def test_packmol_water_num_mols():
    # defining water properties
    species_mol_fractions = np.array([1.0])
    species_molar_masses = np.array([18.01528])  # g/mol
    species_mass_densities = np.array([996.59])  # kg/m3
    pm_shape = "box"
    pm_length_scale = 10.0

    num_mols = get_num_mols(
        species_mol_fractions,
        species_molar_masses,
        species_mass_densities,
        pm_shape,
        pm_length_scale,
    )
    assert np.allclose(num_mols, np.array([33]))
