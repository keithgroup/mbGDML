# MIT License
#
# Copyright (c) 2020-2023, Alex M. Maldonado
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
from ..logger import GDMLLogger

log = GDMLLogger(__name__)


# pylint: disable-next=invalid-name
class mbGDMLData:
    r"""Parent class for mbGDML structure, data, and predict sets."""

    def __init__(self):
        pass

    @property
    def Z(self):
        r"""Atomic numbers of all atoms in data set structures.

        A ``(n,)`` shape array of type :obj:`numpy.int32` containing atomic
        numbers of atoms in the structures in order as they appear.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, "_Z"):
            return self._Z

        return np.array([], dtype=np.int32)

    @Z.setter
    def Z(self, var):
        self._Z = var  # pylint: disable=invalid-name

    @property
    def n_Z(self):
        r"""Number of atoms.

        :type: :obj:`int`
        """
        if self.Z.ndim in (1, 2):
            return int(self.Z.shape[0])

        return int(self.Z.shape[1])

    @property
    def R(self):
        r"""Atomic coordinates of structure(s).

        A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is the
        number of structures and ``n`` is the number of atoms with three
        Cartesian components.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, "_R"):
            return self._R

        return np.array([[[]]])

    @R.setter
    def R(self, var):
        if var.ndim == 2:
            var = np.array([var])
        self._R = var  # pylint: disable=invalid-name

    @property
    def n_R(self):
        r"""Number of structures.

        :type: :obj:`int`
        """
        if self.R.ndim == 2:
            return 1

        return int(self.R.shape[0])

    @property
    def r_unit(self):
        r"""Units of distance. Options are ``'Angstrom'`` or ``'bohr'``.

        :type: :obj:`str`
        """
        return self._r_unit

    @r_unit.setter
    def r_unit(self, var):
        self._r_unit = var

    def save(self, file_path, data):
        r"""General save function for GDML data sets and models.

        Parameters
        ----------
        file_path : :obj:`str`
            Path to save the file.
        data : :obj:`dict`
            Data to be saved to ``npz`` file.
        """
        np.savez_compressed(file_path, **data)
