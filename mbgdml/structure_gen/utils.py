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

"""Structure generation utilities"""

import numpy as np
from ..logger import GDMLLogger

log = GDMLLogger(__name__)

NA = 6.02214076e23  # particles/mole


def mol_to_mass_fractions(mol_fractions, molar_masses):
    r"""Computes the mass fraction of all species present.

    Parameters
    ----------
    mole_fractions : :obj:`numpy.ndarray`, ndim: ``1``
        Mole fractions of each species in the system.
    molar_masses : :obj:`numpy.ndarray`, ndim: ``1``
        Molar mass fo each species in g/mol.

    Returns
    -------
    :obj:`numpy.ndarray`
        Mass fractions of each species.
    """
    mass_species = mol_fractions * molar_masses
    return mass_species / np.sum(mass_species)


def get_mixture_density(mass_fractions, mass_densities):
    r"""Estimate mass density of mixture.

    .. math::

        x_\text{mixture} = \sum x_i \rho_i

    Where :math:`x_i` and :math:`\rho_i` are the mass fraction and density of species
    :math:`i`.

    Parameters
    ----------
    mass_fractions : :obj:`numpy.ndarray`
        Mass fractions of each species in the system.
    mass_densities : :obj:`numpy.ndarray`
        Mass densities of each species in kg/m3.

    Returns
    -------
    :obj:`float`
        Mixture mass density.
    """
    return np.sum(mass_fractions * mass_densities)


def get_shape_volume(shape, length_scale):
    r"""Calculate the volume of the system in m3.

    Parameters
    ----------
    shape : :obj:`str`
        Desired packmol shape. Supported options: ``sphere``, ``box``.
    length_scale : :obj:`float`
        Relevant length scale in Angstroms for the packmol shape.

        - ``sphere``: diameter;
        - ``box``: side length.
    """
    length_scale *= 1e-10  # Ang -> m
    if shape == "sphere":
        radius = length_scale / 2
        volume = (4 / 3) * np.pi * radius**3
    elif shape == "box":
        volume = length_scale**3
    else:
        raise ValueError("Not a valid shape selection.")
    return volume


def get_num_mols(mol_fractions, molar_masses, mass_densities, shape, length_scale):
    r"""Compute the number of molecules for each species given their mole fraction and
    mass densities.

    Parameters
    ----------
    mol_fractions : :obj:`numpy.ndarray`, ndim: ``1``
        Mole fractions of all species.
    molar_masses : :obj:`numpy.ndarray`, ndim: ``1``
        Molar masses of each species in g/mol.
    mass_densities : :obj:`numpy.ndarray`, ndim: ``1``
        Mass density of the all species in kg/m3.
    shape : :obj:`str`
        Desired packmol shape. Supported options: ``sphere``, ``box``.
    length_scale : :obj:`float`
        Relevant length scale in Angstroms for the packmol shape.

        - ``sphere``: diameter;
        - ``box``: side length.

    Returns
    -------
    :obj:`numpy.ndarray`
        Number of molecules for the specified shape and density.
    """
    mass_fractions = mol_to_mass_fractions(mol_fractions, molar_masses)
    mixture_density = get_mixture_density(mass_fractions, mass_densities)
    volume = get_shape_volume(shape, length_scale)

    mass_total = volume * mixture_density
    mass_of_species = mass_fractions * mass_total
    num_molecules = ((mass_of_species * 1000) / molar_masses) * NA
    return num_molecules.astype(int)
