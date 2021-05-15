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

import itertools
import numpy as np
from numpy.core.arrayprint import dtype_short_repr

def _calc_distance(r1, r2):
    """Calculates the Euclidean distance between two points.

    Parameters
    ----------
    r1 : :obj:`numpy.ndarray`
        Cartesian coordinates of a point with shape ``(3,)``.
    r2 : :obj:`numpy.ndarray`
        Cartesian coordinates of a point with shape ``(3,)``.
    """
    return np.linalg.norm(r1 - r2)

def distance_all(z, R, z_slice, cutoff=None):
    """If the distance between all molecules is less than the cutoff.

    Distances larger than the cutoff will return ``False``, meaning they
    should not be included in the data set.

    All criteria functions should have the same parameters.

    Parameters
    ----------
    z : :obj:`numpy.ndarray`
        A ``(n,)`` shape array of type :obj:`numpy.int32` containing atomic
        numbers of atoms in the structures in order as they appear.
    R : :obj:`numpy.ndarray`
        A :obj:`numpy.ndarray` with shape of ``(n, 3)`` where ``n`` is the
        number of atoms with three Cartesian components.
    z_slice : :obj:`numpy.ndarray`
        Indices of the atoms to be used for the cutoff calculation.
    cutoff : :obj:`list` [:obj:`float`]
        Distance cutoff between the atoms selected by ``z_slice``. Must be
        in the same units (e.g., Angstrom) as ``R``.

    Returns
    -------
    :obj:`bool`
        If the distance between the two dimers is less than the cutoff.
    :obj:`None`
        No distance metric is useful for this.
    """
    if cutoff is not None:
        if len(cutoff) != 1:
            raise ValueError('Only one distance can be provided.')

    accept_r = None
    all_pairs = list(itertools.combinations(z_slice, 2))

    # Checks if any pair of molecules is farther away than the cutoff.
    for pair in all_pairs:
        distance = _calc_distance(R[pair[0]], R[pair[1]])
        if cutoff is not None:
            if distance > cutoff[0]:
                accept_r = False
    accept_r = True
    # All pairs of molecules are within the cutoff.
    return accept_r, None

def distance_sum(z, R, z_slice, cutoff=None):
    """If the sum of pairwise distances from the cluster center is less than
    the cutoff.

    Distances larger than the cutoff will return ``False``, meaning they
    should not be included in the data set.

    All criteria functions should have the same parameters.

    Parameters
    ----------
    z : :obj:`numpy.ndarray`
        A ``(n,)`` shape array of type :obj:`numpy.int32` containing atomic
        numbers of atoms in the structures in order as they appear.
    R : :obj:`numpy.ndarray`
        A :obj:`numpy.ndarray` with shape of ``(n, 3)`` where ``n`` is the
        number of atoms with three Cartesian components.
    z_slice : :obj:`numpy.ndarray`
        Indices of the atoms to be used for the cutoff calculation.
    cutoff : :obj:`list` [:obj:`float`], optional
        Distance cutoff between the atoms selected by ``z_slice``. Must be
        in the same units (e.g., Angstrom) as ``R``.

    Returns
    -------
    :obj:`bool`
        If the distance between the two dimers is less than the cutoff.
        ``None`` if no cutoff is provided.
    :obj:`float`
        Calculated distance metric.
    """
    if cutoff is not None:
        if len(cutoff) != 1:
            raise ValueError('Only one distance can be provided.')

    accept_r = None
    center = np.mean(R, axis=0)

    d_sum = 0.0
    # Calculates distance from center
    for z_i in z_slice:
        d_sum += _calc_distance(center, R[z_i])
    if cutoff is not None:
        if d_sum > cutoff[0]:
            accept_r = False
        else:
            accept_r = True
    
    return accept_r, d_sum
