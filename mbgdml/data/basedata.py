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
from .. import utils

class mbGDMLData():
    """Parent class for mbGDML structure, data, and predict sets.
    """

    def __init__(self):
        pass
    
    @property
    def z(self):
        """Atomic numbers of all atoms in data set structures.
        
        A ``(n,)`` shape array of type :obj:`numpy.int32` containing atomic
        numbers of atoms in the structures in order as they appear.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_z'):
            return self._z
        else:
            return np.array([])
    
    @z.setter
    def z(self, var):
        self._z = var
    
    @property
    def n_z(self):
        """Number of atoms.

        :type: :obj:`int`
        """
        if self.z.ndim == 1 or self.z.ndim == 2:
            return int(self.z.shape[0])
        elif self.z.ndim == 3:
            return int(self.z.shape[1])

    @property
    def R(self):
        """Atomic coordinates of structure(s).
        
        A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is the
        number of structures and ``n`` is the number of atoms with three 
        Cartesian components.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_R'):
            return self._R
        else:
            return np.array([[[]]])

    @R.setter
    def R(self, var):
        if var.ndim == 2:
            var = np.array([var])
        self._R = var

    @property
    def n_R(self):
        """Number of structures.

        :type: :obj:`int`
        """
        if self.R.ndim == 2:
            return 1
        elif self.R.ndim == 3:
            return int(self.R.shape[0])

    @property
    def r_unit(self):
        """Units of distance. Options are ``'Angstrom'`` or ``'bohr'``.

        :type: :obj:`str`
        """
        return self._r_unit
    
    @r_unit.setter
    def r_unit(self, var):
        self._r_unit = var

    def save(self, name, data, save_dir):
        """General save function for GDML data sets and models.
        
        Parameters
        ----------
        name : :obj:`str`
            Name of the file to be saved not including the ``npz`` extension.
        data : :obj:`dict`
            Data to be saved to ``npz`` file.
        save_dir : :obj:`str`
            Directory to save the file (with or without the ``'/'`` suffix).
        """
        save_dir = utils.norm_path(save_dir)
        save_path = save_dir + name + '.npz'
        np.savez_compressed(save_path, **data)
