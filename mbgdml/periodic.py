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

"""Periodic boundary conditions."""

from ase.geometry import find_mic
import numpy as np
from scipy.spatial.distance import pdist
from .logger import GDMLLogger

log = GDMLLogger(__name__)


class Cell:
    r"""Enables :math:`n`-body predictions under periodic boundary conditions.

    The minimum-image convention (mic) is used to reformat :math:`n`-body
    structures in a form resembling non-periodic structures.
    """

    def __init__(self, cell_v, cutoff=None, pbc=True):
        r"""
        Parameters
        ----------
        cell_v : :obj:`numpy.ndarray`, shape: ``(3, 3)``
            The three cell vectors. For example, a cube of length 9.0 would be
            ``[[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]]``.
        cutoff : :obj:`float`, default: ``None``
            A periodic image interaction cutoff. Must be smaller than half
            the smallest cube length (non-cubic cells might have slightly larger
            cutoffs). Is automatically calculated if this is ``None``.
        pbc : :obj:`list` or :obj:`bool`
            Periodic boundary conditions in x-, y- and z-direction. Default is
            to assume periodic boundaries in all directions
            (i.e., ``pbc=True``).
        """
        self.cell_v = cell_v
        if cutoff is not None:
            self.cutoff = cutoff
        self.pbc = pbc

    def d_mic(self, d, check_cutoff=True):
        r"""Applies the minimum-image convention to distance vectors.

        Also checks that all atomic pairwise distances are less than
        ``self.cutoff``. If any are equal to greater than the cutoff then it
        returns :obj:`None`.

        Parameters
        ----------
        d : :obj:`numpy.ndarray`, ndim: ``2``
            Distances computed within the periodic cell.

        Returns
        -------
        :obj:`numpy.ndarray`
            The minimum image coordinates.
        """
        d_periodic, _ = find_mic(d, self.cell_v, pbc=self.pbc)
        # Check cutoff
        if check_cutoff:
            d_pd = pdist(d_periodic, metric="euclidean")
            if np.any(np.ravel(d_pd >= self.cutoff)):
                return None
        return d_periodic

    def r_mic(self, r):
        r"""Find minimum-image convention coordinates of molecule(s) under
        periodic boundary conditions.

        Creates distance vectors of each atom with respect to the first.
        Then applies the minimum-image convention using ``self.d_mic()``.

        Parameters
        ----------
        r : :obj:`numpy.ndarray`, ndim: ``2``
            Cartesian coordinates of atoms under periodic boundary conditions.

        Returns
        -------
        :obj:`numpy.ndarray`
            Cartesian coordinates of atoms after applying the minimum-image
            convention.
        """
        # Computes the distance from the first atom.
        assert r.ndim == 2
        d = np.subtract(r, r[0, :])
        return self.d_mic(d)

    @property
    def cell_v(self):
        r"""The three cell vectors. For example, a cube of length 9.0 would be
        ``[[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]]``.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, "_cell_v"):
            return self._cell_v

        return None

    @cell_v.setter
    def cell_v(self, var):
        var = np.array(var)
        self._cell_v = var
        # Update the cutoff
        self.cutoff = np.min(np.linalg.norm(var, axis=1)) / 2.0

    @property
    def volume(self):
        r"""Volume of the periodic cell.

        The volume of the parallelepiped described by ``cell_v``
        (:math:`\boldsymbol{v}`) is computed with

        .. math::
            \text{Volume} = (v_1 \times v_2) \cdot v_3.

        """
        vec = self.cell_v
        return np.dot(vec[2], np.cross(vec[0], vec[1]))
