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


import numpy as np
from cclib.io import ccread


class structure:
    """Basics of structures defined by positions of atoms.

    Makes using structures with mbGDML easier by providing a
    simple-to-use class to automate parsing of xyz or calculation files or
    manually specifying atoms and coordinates.

    Methods
    -------
    parse(file_path)
        Parses file using cclib for atomic numbers and xyz coordinates.
    
    Attributes
    ----------
    z : numpy.ndarray
        A (n,) shape array of type numpy.int32 containing atomic numbers of
        atoms in the structures in order as they appear.
    R : numpy.ndarray
        The atomic positions of structure(s) in a np.ndarray. Has a shape of
        (n, m, 3) where n is the number of structures and m is the number of
        atoms with 3 positional coordinates.
    z_num : int
        The number of atoms stored in z.
    R_num : int
        The number of structures stored in R.
    """

    def __init__(self):
        pass


    def parse(self, file_path):
        """Parses file using cclib for atomic numbers and xyz coordinates.

        Parameters
        ----------
        file_path : str
            Path to file parsable by cclib.
        """

        self._ccdata = ccread(file_path)
        self.z = self._ccdata.atomnos
        self.R = self._ccdata.atomcoords
    

    @property
    def z(self):
        """Array of atoms' atomic numbers in the same number as in
        the coordinate file.
        """

        return self._z
    

    @z.setter
    def z(self, atoms):
        """Ensures z has type numpy.ndarray

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
        """The atomic positions of structure(s) in a np.ndarray.
        
        R has a shape of (n, m, 3) where n is the number of structures and
        m is the number of atoms with 3 positional coordinates.
        """

        return self._R
    

    @R.setter
    def R(self, coords):
        """Standardizes R array to always have (n, m, 3) shape.

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
    
