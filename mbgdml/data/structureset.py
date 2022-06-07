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

import os
import numpy as np
from cclib.parser.utils import convertor
from .basedata import mbGDMLData
from .. import __version__
from ..parse import parse_stringfile
from .. import utils

class structureSet(mbGDMLData):
    """For creating, loading, manipulating, and using structure sets.

    Attributes
    ----------
    name : :obj:`str`
        Name of the structure set. Defaults to ``'structureset'``.
    """

    def __init__(self, *args):
        """
        Parameters
        ----------
        structureset_path : :obj:`str`, optional
            Path to a saved `npz`.
        """
        self.type = 's'
        self.name = 'structureset'
        if len(args) == 1:
            self.load(args[0])
    
    @property
    def entity_ids(self):
        """1D array specifying which atoms belong to which entities.
        
        An entity represents a related set of atoms such as a single molecule,
        several molecules, or a functional group. For mbGDML, an entity usually
        corresponds to a model trained to predict energies and forces of those
        atoms. Each ``entity_id`` is an :obj:`int` starting from ``0``.
        
        It is conceptually similar to PDBx/mmCIF ``_atom_site.label_entity_ids``
        data item.

        Examples
        --------
        A single water molecule would be ``[0, 0, 0]``. A water (three atoms)
        and methanol (six atoms) molecule in the same structure would be 
        ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_entity_ids'):
            return self._entity_ids
        else:
            return np.array([])
    
    @entity_ids.setter
    def entity_ids(self, var):
        self._entity_ids = var
    
    @property
    def comp_ids(self):
        """1D array relating ``entity_id`` to a fragment label for chemical
        components or species. Labels could be ``WAT`` or ``h2o`` for water,
        ``MeOH`` for methanol, ``bz`` for benzene, etc. There are no
        standardized labels for species. The index of the label is the
        respective ``entity_id``. For example, a water and methanol molecule
        could be ``['h2o', 'meoh']``.

        Examples
        --------
        Suppose we have a structure containing a water and methanol molecule.
        We can use the labels of ``h2o`` and ``meoh`` (which could be
        anything): ``['h2o', 'meoh']``.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_comp_ids'):
            return self._comp_ids
        else:
            return np.array([])
    
    @comp_ids.setter
    def comp_ids(self, var):
        self._comp_ids = var

    @property
    def md5(self):
        """Unique MD5 hash of structure set.

        Notes
        -----
        :obj:`mbgdml.data.basedata.mbGDMLData.z` and
        :obj:`mbgdml.data.basedata.mbGDMLData.R` are used to generate the MD5
        hash.

        :type: :obj:`str`
        """
        return utils.md5_data(self.asdict(), ['z', 'R'])
    
    def convertR(self, R_units):
        """Convert coordinates and updates ``r_unit``.

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
        self.entity_ids = structureset['entity_ids']
        self.comp_ids = structureset['comp_ids']
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

    def from_xyz(self, file_path, r_unit, entity_ids, comp_ids):
        """Reads data from xyz files and sets attributes.

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
        entity_ids : :obj:`numpy.ndarray`
            A uniquely identifying integer specifying what atoms belong to
            which entities. Entities can be a related set of atoms, molecules,
            or functional group. For example, a water and methanol molecule
            could be ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.
        comp_ids : :obj:`numpy.ndarray`
            Relates ``entity_id`` to a fragment label for chemical components
            or species. Labels could be ``WAT`` or ``h2o`` for water, ``MeOH``
            for methanol, ``bz`` for benzene, etc. There are no standardized
            labels for species. The index of the label is the respective
            ``entity_id``. For example, a water and methanol molecule could
            be ``['h2o', 'meoh']``.

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
        self.entity_ids = np.array(entity_ids)
        self.comp_ids = np.array(comp_ids)
    
    def from_npz(self, file_path, z_label, R_label, r_unit,
        entity_ids, comp_ids):
        """Reads data from npz file and sets attributes.

        Parameters
        ----------
        file_path : :obj:`str`
            Path to npz file.
        z_label : :obj:`str`
            Npz label that contains the z data as an array.
        R_label : :obj:`str`
            Npz label that contains the R data as an array.
        r_unit : :obj:`str`
            Units of distance. Options are ``'Angstrom'`` or ``'bohr'`` (defined
            by cclib).
        entity_ids : :obj:`numpy.ndarray`
            A uniquely identifying integer specifying what atoms belong to
            which entities. Entities can be a related set of atoms, molecules,
            or functional group. For example, a water and methanol molecule
            could be ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.
        comp_ids : :obj:`numpy.ndarray`
            Relates ``entity_id`` to a fragment label for chemical components
            or species. Labels could be ``WAT`` or ``h2o`` for water, ``MeOH``
            for methanol, ``bz`` for benzene, etc. There are no standardized
            labels for species. The index of the label is the respective
            ``entity_id``. For example, a water and methanol molecule could
            be ``['h2o', 'meoh']``.

        """
        self.name = os.path.splitext(os.path.basename(file_path))[0]
        npz_data = dict(np.load(file_path))
        
        self.z = npz_data[z_label]
        self.R = npz_data[R_label]
        self.r_unit = r_unit
        self.entity_ids = np.array(entity_ids)
        self.comp_ids = np.array(comp_ids)


    def asdict(self):
        """Converts object into a custom :obj:`dict`.

        Returns
        -------
        :obj:`dict`
        """
        structureset = {
            'type': np.array('s'),
            'mbgdml_version': np.array(__version__),
            'name': np.array(self.name),
            'z': np.array(self.z),
            'R': np.array(self.R),
            'r_unit': np.array(self.r_unit),
            'entity_ids': self.entity_ids,
            'comp_ids': self.comp_ids
        }

        if len(structureset['z']) != len(structureset['entity_ids']):
            raise ValueError('Number of atoms in z and entity_ids is not the same.')

        structureset['md5'] = np.array(utils.md5_data(structureset, ['z', 'R']))
        return structureset
    
    def write_xyz(self, save_dir):
        """Saves xyz file of all structures in data set.

        Parameters
        ----------
        save_dir : :obj:`str`
        """
        xyz_path = os.path.join(save_dir, self.name)
        utils.write_xyz(xyz_path, self.z, self.R)
