# MIT License
# 
# Copyright (c) 2020, Alex M. Maldonado
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

from mbgdml.parse import parse_stringfile,parse_cluster
from mbgdml.partition import partition_cluster
from mbgdml.solvents import system_info
from mbgdml.utils import atoms_by_number, string_coords
import numpy as np
from cclib.io import ccread

class structure:
    """Basics of structures defined by positions of atoms.

    Makes using structures with mbGDML easier by providing a
    simple-to-use class to automate parsing of xyz or calculation files or
    manually specifying atoms and coordinates.

    Parameters
    ----------
    file_path : :obj:`str`
        Path to xyz file.
    """

    def __init__(self, *args):
        if len(args) == 1:
            self.parse(args[0])

    def parse(self, file_path):
        """Parses file using cclib for atomic numbers and xyz coordinates.

        Parameters
        ----------
        file_path : :obj:`str`
            Path to xyz file.
        """
        try:
            self._ccdata = ccread(file_path)
            self._z = self._ccdata.atomnos
            if np.shape(self._ccdata.atomcoords)[0] == 1:
                self._R = self._ccdata.atomcoords[0]
            else:
                raise ValueError()
        except:
            z, _, data = parse_stringfile(file_path)
            self._z = np.array(atoms_by_number(z[0]))
            if len(data) == 1:
                self._R = np.array(data[0])
            else:
                raise ValueError(
                    f'{file_path} contains more than one structure.'
                )

    @property
    def z(self):
        """Atomic numbers of all atoms.
        
        A ``(n,)`` shape array of type :obj:`numpy.int32` containing atomic
        numbers of atoms in the structures in order as they appear.

        :type: :obj:`numpy.ndarray`
        """
        return self._z
    
    @z.setter
    def z(self, atoms):
        """Ensures ``z`` has type :obj:`numpy.ndarray`

        Raises
        ------
        TypeError
            Coords type should be numpy.ndarray.
        ValueError
            Coordinates should have at least two dimensions but not more than
            three.
        """
        if type(atoms) != np.ndarray:
            raise TypeError(
                f'Atomic coordinates must be in np.ndarray format'
            )
        self._z = atoms

    @property
    def R(self):
        """Atomic coordinates of structure(s).
        
        A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is the
        number of structures and ``n`` is the number of atoms with three 
        Cartesian components.

        :type: :obj:`numpy.ndarray`
        """
        return self._R
    
    @R.setter
    def R(self, coords):
        """Standardizes ``R`` array to always have (n, m, 3) shape.

        Raises
        ------
        TypeError
            Coords type should be numpy.ndarray.
        ValueError
            Coordinates should have at least two dimensions but not more than
            three.
        """
        if type(coords) != np.ndarray:
            raise TypeError(
                f'Atomic coordinates must be in numpy.ndarray format'
            )

        if coords.ndim == 2:
            self._R = np.array([coords])
        elif coords.ndim == 3:
            self._R = coords
        else:
            raise ValueError(f'Unusual R dimension of {coords.ndim}; '
                              'should be two or three.')
    
    @property
    def z_num(self):
        """The number of atoms.
        """
        return self._z.shape[0]

    @property
    def R_num(self):
        """The number of structures.
        """
        if self.R.ndim == 2:
            return 1
        elif self.R.ndim == 3:
            if self.R.shape[0] == 1:
                return 1
            else:
                return int(self.R.shape[0])
        else:
            raise ValueError(
                f"The coordinates have an unusual dimension of {self.R.ndim}."
            )
    
    @property
    def molecules(self):
        """Molecules in structure.

        :obj:`int` keys specifying the molecule number starting from 0 with 
        nested :obj:`dict` as values which contain

        ``'z'``
            A ``(n,)`` shape array of type :obj:`numpy.int32` containing
            atomic numbers of atoms in the structures in order as they
            appear.
        
        ``'R'``
            A :obj:`numpy.ndarray` with shape of ``(n, 3)`` where ``n`` is
            the number of atoms with three Cartesian components.
        
        :type: :obj:`dict`
        """
        return parse_cluster(self.z, self.R)
    
    def partitions(self, max_nbody=4):
        """Partitions of the structure.

        Parameters
        ----------
        max_nbody: :obj:`int`, optional
            Highest order of n-body structure to include.
        
        Returns
        -------
        :obj:`dict`
            All nonrepeating molecule combinations from original cluster with
            keys being uppercase concatenations of molecule labels and values
            being the string of coordinates.
        """
    
        sys_info = system_info(self.z.tolist())
        all_partitions = {}  
        
        # Loops through all possible n-body partitions and adds the atoms
        # once and adds each step traj_partition.
        i_nbody = 1
        while i_nbody <= sys_info['cluster_size'] and i_nbody <= max_nbody:
            partitions = partition_cluster(self.molecules, i_nbody)
            partition_labels = list(partitions.keys())
            # Tries to add the next trajectory step to 'coords'; if it fails it
            # initializes 'atoms' and 'coords' for that partition.
            for label in partition_labels:
                partition_info = system_info(
                    partitions[label]['z'].tolist()
                )

                try:
                    all_partitions[label]['R'] = np.append(
                        all_partitions[label]['R'],
                        np.array([partitions[label]['R']]),
                        axis=0
                    )
                except KeyError:
                    all_partitions[label] = {
                        'solvent_label': partition_info['solvent_label'],
                        'cluster_size': sys_info['cluster_size'],
                        'partition_label': label,
                        'partition_size': partition_info['cluster_size'],
                        'z': partitions[label]['z'],
                        'R': np.array([partitions[label]['R']])
                    }
            i_nbody += 1
        return all_partitions

    def write_xyz(self, name, save_dir='./'):
        """Writes xyz file of structure

        Parameters
        ----------
        name : :obj:`str`
            File name.
        save_dir : :obj:`str`
            Path to directory to save file.
        """
        if self.R.ndim == 3 and np.shape(self.R)[0] == 1:
            R_string = string_coords(self.z, self.R[0])
        elif self.R.ndim == 2:
            R_string = string_coords(self.z, self.R)
        else:
            raise ValueError('More than one structure.')
        if save_dir[-1] != '/':
            save_dir += '/'
        with open(f'{save_dir}{name}.xyz', 'w') as f:
            f.writelines(
                [f'{len(self.z)}\n','\n', R_string]
            )

