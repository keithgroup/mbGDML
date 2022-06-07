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

import numpy as np
from cclib.io import ccread
from cclib.parser.utils import convertor
from ..utils import convert_forces


class PartitionOutput:
    """Quantum chemistry output file for all MD steps of a single partition.

    Output file that contains electronic energies and gradients of the same
    partition from a single MD trajectory. For a single dimer partition of a
    n step MD trajectory would have n coordinates, single point energies,
    and gradients.

    Parameters
    ----------
    output_path : :obj:`str`
        Path to computational chemistry output file that contains energies
        and gradients (preferably ab initio) of all MD steps of a
        single partition.
    cluster_label : :obj:`str`
        Identifies the cluster where this partition originates from. Specifies
        at least the solvent and size of the parent cluster. For example, if
        this partition is the number 23 structure from ABCluster
        for a four water cluster, this could be ``'4H2O.abc23'``.
    partition_label : :obj:`str`
        Specifies the molecules composing this partition. Typically we use a
        single uppercase letter per molecule in the order it appears in the
        xyz coordinates. For example, a partition from a 4H2O cluster with the
        first and third water molecules would be ``'AC'``.
    md_temp : :obj:`int`
        The set point for the MD thermostat. This only affects labeling and
        directory organization as it has a significant impact on geometry
        sampling.
    md_iter : :obj:`int`, optional
        Specifies the MD iteration for labeling and data-tracking purposes.
        Defaults to ``0``.
    e_unit : :obj:`str`, optional
        Desired units of energy. Available units implemented are ``'eV'``,
        ``'hartree'``, ``'kcal/mol'``, and ``'kJ/mol'``. Defaults to
        ``'kcal/mol'``.
    r_unit : :obj:`str`, optional
        Desired units of distance. Available units implemented in cclib's
        convertor function are ``'Angstrom'`` and ``'bohr'``.
    theory : :obj:`str`, optional
        Level of theory for the calculations. For example, ``'mp2.def2tzvp'``.
        Defaults to ``'unknown'``.

    Notes
    -----
    Typical GDML units are kcal/mol and kcal/mol/Angstrom for energies
    and forces, respectively.

    Attributes
    ----------
    output_path : :obj:`str`
        Path to computational chemistry output file that contains energies
        and gradients (preferably ab initio) of all MD steps of a
        single partition.
    cluster_label : :obj:`str`
        Identifies the cluster where this partition originates from. Specifies
        at least the solvent and size of the parent cluster. For example, if
        this partition is the number 23 structure from ABCluster
        for a four water cluster, this could be '4H2O.abc23'.
    partition_label : :obj:`str`
        Specifies the molecules composing this partition. Typically we use a
        single uppercase letter per molecule in the order it appears in the
        xyz coordinates. For example, a partition from a 4H2O cluster with the
        first and third water molecules would be 'AC'.
    md_temp : :obj:`int`
        The set point for the MD thermostat. This only affects labeling and
        directory organization as it has a significant impact on geometry
        sampling.
    md_iter : :obj:`int`, optional
        Specifies the MD iteration for labeling and data-tracking purposes.
        Defaults to 0.
    e_unit : :obj:`str`, optional
        Desired units of energy. Available units implemented are 'eV',
        'hartree', 'kcal/mol', and 'kJ/mol'. Defaults to 'kcal/mol'.
    r_unit : :obj:`str`, optional
        Desired units of distance. Available units implemented in cclib's
        convertor function are 'Angstrom' and 'bohr'.
    output_name : :obj:`str`
        The name of the quantum chemistry output file (no extension).
    partition_size : :obj:`int`
        The number of solvent molecules in the partition.
    cclib_data : `cclib.parser.data.ccData``
        Contains all data parsed from output file.
    z : :obj:`numpy.ndarray`
        A (n,) array containing n atomic numbers.
    R : :obj:`numpy.ndarray`
        A (m, n, 3) array containing the atomic coordinates
        of n atoms of m MD steps.
    E : :obj:`numpy.ndarray`
        A (m,) array containing the energies of m structures.
    G : :obj:`numpy.ndarray`
        A (m, n, 3) array containing the atomic gradients of n atoms of m MD
        steps.
    F : :obj:`numpy.ndarray`
        A (m, n, 3) array containing the atomic forces
        of n atoms of m MD steps. Simply the negative of grads.
    system : :obj:`str`
        From mbgdml.solvents and designates the system. Currently
        only 'solvent' is implemented.
    solvent_info : :obj:`dict`
        If the system is 'solvent', contains information
        about the system and solvent. 'solvent_name', 'solvent_label',
        'solvent_molec_size', and 'cluster_size'.
    """


    def __init__(
        self, output_path, cluster_label, partition_label, md_temp, grad_e_unit,
        grad_r_unit, md_iter=0, e_unit='kcal/mol', r_unit='Angstrom',
        theory='unknown'
    ):
        
        self.output_path = output_path
        self.output_name = self.output_path.split('/')[-1].split('.')[0]
        self.cluster_label = cluster_label
        self.partition_label = partition_label
        self.partition_size = int(len(self.partition_label))
        self.md_temp = md_temp
        self.md_iter = md_iter
        self.e_unit = e_unit
        self.r_unit = r_unit
        self.theory = theory
        
        self.cclib_data = ccread(self.output_path)
        self._get_gdml_data()
        self.E = convertor(self.E, 'eV', self.e_unit)
        self.G = convert_forces(
            self.G, grad_e_unit, grad_r_unit, self.e_unit, self.r_unit
        )
        self.F = np.negative(self.G)

    
    def _get_gdml_data(self):
        """Parses GDML-relevant data from partition output file.

        Raises
        ------
        AttributeError
            There were no parse mpenergies or scfenergies attributes.
        AttributeError
            Not all attributes (atomnos, atomcoords, grads) were
            parsed correctly.
        """

        try:
            self.z = self.cclib_data.atomnos
            self.R = self.cclib_data.atomcoords
            self.G = self.cclib_data.grads
            self.F = np.negative(self.G)

            if hasattr(self.cclib_data, 'mpenergies'):
                self.E = self.cclib_data.mpenergies
            elif hasattr(self.cclib_data, 'scfenergies'):
                self.E = self.cclib_data.scfenergies
            else:
                raise AttributeError('No energies were found.')
        except BaseException:
            raise AttributeError(
                f'Some attributes (atomnos, atomcoords, grads) were'
                f'not found in {self.output_name}'
            )
