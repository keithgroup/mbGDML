# MIT License
# 
# Copyright (c) 2022, Alex M. Maldonado
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

"""Periodic boundary conditions."""

from ase.geometry import find_mic
import numpy as np
from scipy.spatial.distance import pdist

class Cell:
    """Define a periodic cell for many-body expansions.

    This is mostly a wrapper for ``ase.geometry.find_mic``.
    """

    def __init__(self, cell_v, cutoff, pbc=True):
        """
        Parameters
        ----------
        cell_v : :obj:`numpy.ndarray`, shape: ``(3, 3)``
            The three cell vectors. For example, a cube of length 9.0 would be
            ``[[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]]``.
        cutoff : :obj:`float`
            A periodic image interaction cutoff. Must not be larger than half
            the cube length (non-cubic cells might have slightly larger
            cutoffs).
        pbc : :obj:`list` or :obj:`bool`
            Periodic boundary conditions in x-, y- and z-direction. Default is
            to assume periodic boundaries in all directions
            (i.e., ``pbc=True``).
        """
        self.cell_v = cell_v
        self.cutoff = cutoff
        self.pbc = pbc

    def r_mic(self, r):
        """Find minimum image convention coordinates of atoms in a periodic
        cell.

        Parameters
        ----------
        r : :obj:`numpy.ndarray`
            Cartesian coordinates of atoms under periodic boundary conditions.
        """
        r = np.subtract(r, r[0,:])
        r_periodic, _ = find_mic(r, self.cell_v, pbc=self.pbc)
        # Check cutoff
        r_pd = pdist(r_periodic, metric='euclidean')
        if np.any(np.ravel(r_pd >= self.cutoff)):
            return None
        else:
            return r_periodic
