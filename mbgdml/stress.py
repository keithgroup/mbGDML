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

"""Compute stress"""

import numpy as np
from .periodic import Cell
from .logger import GDMLLogger

log = GDMLLogger(__name__)

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
    r"""Convert array to Voigt notation.

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


def virial_finite_diff(
    Z,
    R,
    entity_ids,
    comp_ids,
    cell_vectors,
    mbe_pred,
    dh=1e-4,
    only_normal_stress=False,
):
    r"""Approximates the virial stress of a periodic cell with vectors
    :math:`\mathbf{h}` using a finite difference scheme.

    .. math::

        W_{ij} = \frac{E_{h_{ij} + \Delta h} - E_{h_{ij} - \Delta h}}{2 \Delta h}

    Parameters
    ----------
    Z : :obj:`numpy.ndarray`, ndim: ``1``
        Atomic numbers of all atoms in the system with respect to ``R``.
    R : :obj:`numpy.ndarray`, shape: ``(N, len(Z), 3)``
        Cartesian coordinates of ``N`` structures to predict.
    entity_ids : :obj:`numpy.ndarray`, shape: ``(N,)``
        Integers specifying which atoms belong to which entities.
    comp_ids : :obj:`numpy.ndarray`, shape: ``(N,)``
        Relates each ``entity_id`` to a fragment label. Each item's index
        is the label's ``entity_id``.
    cell_vectors : :obj:`numpy.ndarray`, shape: ``(3, 3)``
        The three cell vectors.
    mbe_pred : :obj:`mbgdml.mbe.mbePredict`
        Initialized many-body expansion predictor.
    dh : :obj:`float`, default: ``1e-4``
        Forward and backward displacement for the cell vectors.
    only_normal_stress : :obj:`bool`
        Only compute normal (xx, yy, and zz) stress.

    Returns
    -------
    :obj:`numpy.ndarray`
        (shape: ``(3, 3)``) - Virial stress in units of energy.

    Notes
    -----
    Code adapted from `here
    <https://github.com/svandenhaute/openyaff/blob/main/openyaff/utils.py#L442>`__.
    """
    # pylint: disable=invalid-name
    compute_stress_original = mbe_pred.compute_stress
    mbe_pred.compute_stress = False  # Prevents recursion
    R_fractional = np.dot(R, np.linalg.inv(cell_vectors))
    dEdh = np.zeros((3, 3))

    for i in range(0, 3):
        for j in range(0, 3):
            if only_normal_stress and (i != j):
                continue

            cell_vectors_ = cell_vectors.copy()

            # Backward
            cell_vectors_[i, j] -= dh / 2
            R_tmp = np.dot(R_fractional, cell_vectors_)
            cell_vectors_tmp = cell_vectors_.copy()
            mbe_pred.periodic_cell = Cell(cell_vectors_tmp, pbc=True)
            E_minus, _ = mbe_pred.predict(Z, R_tmp, entity_ids, comp_ids)

            # Forward
            cell_vectors_[i, j] += dh
            R_tmp = np.dot(R_fractional, cell_vectors_)
            cell_vectors_tmp = cell_vectors_.copy()
            mbe_pred.periodic_cell = Cell(cell_vectors_tmp, pbc=True)
            E_plus, _ = mbe_pred.predict(Z, R_tmp, entity_ids, comp_ids)
            dEdh[i, j] = (E_plus - E_minus) / dh

    # Puts the original value of compute_stress back.
    mbe_pred.compute_stress = compute_stress_original
    return dEdh
