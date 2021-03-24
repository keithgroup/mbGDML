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

import os
import numpy as np
from cclib.parser.utils import convertor
from mbgdml.data import mbGDMLData
from mbgdml import __version__
from mbgdml.parse import parse_stringfile
from mbgdml import utils

class structureSet(mbGDMLData):
    """For creating, loading, manipulating, and using structure sets.

    Parameters
    ----------
    structureset_path : :obj:`str`, optional
        Path to a saved :obj:`numpy.NpzFile`.

    Attributes
    ----------
    name : :obj:`str`
        Name of the structure set. Defaults to ``'structureset'``.
    """

    def __init__(self, *args):
        self.type = 's'
        self.name = 'structureset'
        if len(args) == 1:
            self.load(args[0])
    
    @property
    def mol_ids(self):
        """An array specifying which atoms belong to what molecules.

        All atoms should be grouped together.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_mol_ids'):
            return self._mol_ids
        else:
            return None
    
    @mol_ids.setter
    def mol_ids(self, var):
        self._mol_ids = var

    @property
    def md5(self):
        """Unique MD5 hash of structure set.

        :type: :obj:`bytes`
        """
        return utils.md5_data(self.structureset, ['z', 'R'])
    
    def convertR(self, R_units):
        """Convert coordinates and updates :attr:`r_unit`.

        Parameters
        ----------
        R_units : :obj:`str`
            Desired units of coordinates. Options are ``'Angstrom'`` or
            ``'bohr'``.
        """
        self._R = convertor(self.R, self.r_unit, R_units)
        self.r_unit = R_units
    
    def _update(self, structureset):
        """Updates object attributes.

        Parameters
        ----------
        structureset : :obj:`dict`
            Contains all information and arrays stored in data set.
        """
        self.name = str(structureset['name'][()])
        self.mol_ids = structureset['mol_ids']
        self._z = structureset['z']
        self._R = structureset['R']
        self._r_unit = str(structureset['r_unit'][()])
        self.mbgdml_version = str(structureset['mbgdml_version'][()])

    def load(self, structureset_path):
        """Read data set.

        Parameters
        ----------
        structureset_path : :obj:`str`
            Path to NumPy ``npz`` file.
        """
        structureset_npz = np.load(structureset_path, allow_pickle=True)
        npz_type = str(structureset_npz.f.type[()])
        if npz_type != 's':
            raise ValueError(f'{npz_type} is not a structure set.')
        else:
            self._update(dict(structureset_npz))

    def from_xyz(self, file_path, r_unit, molecule_ids):
        """Reads data from xyz files and sets z and R data.

        If using the extended XYZ format will assume coordinates are the first
        three data columns (after atom symbols).

        Molecule specifications are needed for data set sampling.

        Parameters
        ----------
        file_path : :obj:`str`
            Path to xyz file.
        r_unit : :obj:`str`
            Units of distance. Options are ``'Angstrom'`` or ``'bohr'`` (defined
            by cclib).
        molecule_ids : :obj:`list` [:obj:`int`]
            List of molecule indices starting from ``0``. For example, a water
            dimer would be ``[0, 0, 0, 1, 1, 1]``.
        """
        self.name = os.path.splitext(os.path.basename(file_path))[0]

        z, _, data = parse_stringfile(file_path)
        z = [utils.atoms_by_number(i) for i in z]

        # If all the structures have the same order of atoms (as required by
        # sGDML), condense into a one-dimensional array.
        if len(set(tuple(i) for i in z)) == 1:
            z = np.array(z[0])
        else:
            z = np.array(z)
        self._z = z

        # Stores Cartesian coordinates.
        data = np.array(data)
        if data.shape[2] == 6:
            self._R = data[:,:,3:]
        elif data.shape[2] == 3:
            self._R = data
        else:
            raise ValueError(f'There was an issue parsing R from {file_path}.')
        
        self.r_unit = r_unit
        self.mol_ids = np.array(molecule_ids)

    @property
    def structureset(self):
        """Contains all data as :obj:`numpy.ndarray` objects.

        :type: :obj:`dict`
        """
        structureset = {
            'type': np.array('s'),
            'mbgdml_version': np.array(__version__),
            'name': np.array(self.name),
            'z': np.array(self.z),
            'R': np.array(self.R),
            'r_unit': np.array(self.r_unit),
            'mol_ids': self.mol_ids
        }

        if len(structureset['z']) != len(structureset['mol_ids']):
            raise ValueError('Number of atoms in z and mol_ids is not the same.')

        structureset = self.add_system_info(structureset)
        structureset['md5'] = np.array(utils.md5_data(structureset, ['z', 'R']))
        return structureset
