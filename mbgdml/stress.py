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

"""Compute stress"""

import logging
import numpy as np

log = logging.getLogger(__name__)

VOIGT_INDICES = [[0, 0], [1, 1], [2, 2], [1, 0], [0, 2], [0, 1]]


def virial_atom_loop(R, F):
    r"""Sums the outer tensor products for all atoms:

    .. math::

        \sum_i \mathbf{r}_{i} \otimes \mathbf{F}_{i}.

    Parameters
    ----------
    R : :obj:`numpy.ndarray`, ndim: ``2``
        Cartesian coordinates of all atoms to loop over.
    F : :obj:`numpy.ndarray`, ndim: ``2``
        Forces to use for outer tensor product.

    Returns
    -------
    :obj:`numpy.ndarray`
        (shape: ``(3, 3)``) - Virial contribution.
    """
    virial = np.zeros((3, 3), dtype=np.float64)
    for r_atom, f_atom in zip(R, F):
        virial += np.multiply.outer(r_atom, f_atom)
    return virial


def to_voigt(arr):
    """Convert array to Voigt notation.

    Parameters
    ----------
    arr : :obj:`numpy.ndarray`, shape: ``(3, 3)``
        Virial, stress, pressor, etc. array.

    Returns
    -------
    :obj:`numpy.ndarray`
        Array in Voigt notation.
    """
    return np.array([arr[i, j] for i, j in VOIGT_INDICES])
