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

"""Analyses for models."""

import numpy as np
from ..logger import GDMLLogger

log = GDMLLogger(__name__)


def gdml_mat52_wrk(r_desc, sig, n_perms, R_desc_perms):
    r"""Compute the Matérn 5/2 covariance function for a single structure with
    respect to a GDML train set.

    Parameters
    ----------
    r_desc : :obj:`numpy.ndarray`, ndim: ``1``
        GDML descriptor of a single structure with permutational symmetries
        specified by the model.
    sig : :obj:`float`
        Trained kernel length scale.
    n_perms : :obj:`int`
        Number of permutational symmetries.
    R_desc_perms : :obj:`numpy.ndarray`, ndim: ``2``
        Training descriptors with permutational symmetries.

    Returns
    -------
    :obj:`numpy.ndarray`
        (ndim: ``1``) Covariances between a single structure and the GDML
        training set.

    Notes
    -----
    The Matérn kernel when :math:`\nu = 5/2` reduces to the following
    expression.

    .. math::

        k_{5/2} (x_i, x_j) = \left( 1 + \frac{\sqrt{5}}{l} d(x_i, x_j) \
        + \frac{5}{3l} d(x_i, x_j)^2 \right) \exp \left( - \frac{\sqrt{5}}{l}
        d (x_i, x_j) \right)

    For GDML, sigma (``sig``) is the kernel length-scale parameter :math:`l`.
    :math:`d(x_i, x_j)` is the Euclidean distance between :math:`x_i`
    (``r_desc``) and :math:`x_j` (a single training point in ``R_desc_perms``).

    .. note::

        This can be a ray task if desired.

    """
    n_train = int(R_desc_perms.shape[0] / n_perms)
    dim_c = n_train * n_perms

    # Pre-allocate memory.
    mat52_exp = np.empty((dim_c,), dtype=np.float64)
    mat52 = np.ones((dim_c,), dtype=np.float64)

    # Avoid divisions.
    mat52_base_fact = 5.0 / (3 * sig**2)
    sqrt5 = np.sqrt(5.0)
    diag_scale_fact = sqrt5 / sig

    # Computes the difference between the descriptor and the training set.
    diff_ab_perms = np.subtract(
        np.broadcast_to(r_desc, R_desc_perms.shape), R_desc_perms
    )
    # Computes the norm of descriptor difference.
    norm_ab_perms = np.linalg.norm(diff_ab_perms, ord=None, axis=1)

    # Matern 5/2 exponential term.
    np.exp(-diag_scale_fact * norm_ab_perms, out=mat52_exp)
    mat52 += diag_scale_fact * norm_ab_perms
    mat52 += mat52_base_fact * np.power(norm_ab_perms, 2)
    mat52 *= mat52_exp

    return mat52


def gdml_mat52(model, Z, R):
    r"""Compute the Matérn 5/2 covariance function with respect to a GDML model.

    This is a light wrapper around :func:`mbgdml.analysis.models.gdml_mat52_wrk`
    that can run in parallel.

    Parameters
    ----------
    model : :obj:`mbgdml.models.gdmlModel`
        GDML model containing all information need to make predictions.
    Z : :obj:`numpy.ndarray`, ndim: ``1``
        Atomic numbers of all atoms in ``R`` (in the same order).
    R : :obj:`numpy.ndarray`, ndim: ``2`` or ``3``
        Cartesian coordinates of a single structure to predict.

    Returns
    -------
    :obj:`numpy.ndarray`
        (ndim: ``1`` or ``2``) Covariances between all structures in ``R`` and
        the GDML training set.
    """
    if R.ndim == 2:
        R = R[None, ...]
    n_atoms = len(Z)
    n_R = R.shape[0]
    desc_size = model.R_desc_perms.shape[0]
    desc_func = model.desc_func

    mat52 = np.empty((n_R, desc_size), dtype=np.float64)
    R = R.reshape(n_R, n_atoms * 3)

    for i in range(n_R):
        r_desc, _ = desc_func(R[i], model.lat_and_inv)
        mat52[i] = gdml_mat52_wrk(r_desc, model.sig, model.n_perms, model.R_desc_perms)

    if n_R == 1:
        return mat52[0]

    return mat52
