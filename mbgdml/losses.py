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
from .logger import GDMLLogger

log = GDMLLogger(__name__)


def mae(errors):
    r"""Mean absolute error (MAE).

    .. math::

        MAE = \frac{1}{n} \sum_{i=1}^n \mid \hat{y}_{i} - y_{i} \mid,

    where :math:`y_{i}` is the true value, :math:`\hat{y}_{i}` is the
    predicted value, :math:`\mid \cdots \mid` is the absolute value, and
    :math:`n` is the number of data points.

    Parameters
    ----------
    errors : :obj:`numpy.ndarray`
        Array of :math:`\hat{y} - y` values. This will be flattened beforehand.

    Returns
    -------
    :obj:`float`
        Mean absolute error.

    See Also
    --------
    mse, rmse, sse
    """
    return np.mean(np.abs(errors.flatten()))


def mse(errors):
    r"""Mean squared error (MSE).

    .. math::

        MSE = \frac{1}{n} \sum_{i=1}^n \left( \hat{y}_{i} - y_{i} \right)^2,

    where :math:`y_{i}` is the true value, :math:`\hat{y}_{i}` is the
    predicted value, and :math:`n` is the number of data points.

    Parameters
    ----------
    errors : :obj:`numpy.ndarray`
        Array of :math:`\hat{y} - y` values. This will be flattened beforehand.

    Returns
    -------
    :obj:`float`
        Mean squared error.

    See Also
    --------
    mae, rmse, sse
    """
    return np.mean(errors.flatten() ** 2)


def rmse(errors):
    r"""Root-mean-squared error.

    .. math::

        RMSE = \sqrt{ \frac{\sum_{i=1}^n \left( \hat{y}_{i} - y_{i} \right)^2}{n} },

    where :math:`y_{i}` is the true value, :math:`\hat{y}_{i}` is the
    predicted value, and :math:`n` is the number of data points.

    Parameters
    ----------
    errors : :obj:`numpy.ndarray`
        Array of :math:`\hat{y} - y` values. This will be flattened beforehand.

    Returns
    -------
    :obj:`float`
        Root-mean-squared error.

    See Also
    --------
    mae, mse, sse
    """
    return np.sqrt(mse(errors))


def sse(errors):
    r"""Sum of squared errors.

    .. math::

        SSE = \sum_{i=1}^n \left( \hat{y}_{i} - y_{i} \right)^2,

    where :math:`y_{i}` is the true value, :math:`\hat{y}_{i}` is the
    predicted value, and :math:`n` is the number of data points.

    Parameters
    ----------
    errors : :obj:`numpy.ndarray`
        Array of :math:`\hat{y} - y` values. This will be flattened beforehand.

    Returns
    -------
    :obj:`float`
        Sum of squared errors.

    See Also
    --------
    mae, mse, rmse
    """
    return np.dot(errors.flatten(), errors.flatten())


def loss_f_mse(results):
    r"""Force MSE.

    Parameters
    ----------
    results : :obj:`dict`
        Results which contains at least force errors under the ``'force'`` key.

    Returns
    -------
    :obj:`float`
        Loss.

    See Also
    --------
    loss_f_rmse, loss_f_e_weighted_mse
    """
    return mse(results["force"])


def loss_f_rmse(results):
    r"""Force RMSE.

    Parameters
    ----------
    results : :obj:`dict`
        Results which contains at least force errors under the ``'force'`` key.

    Returns
    -------
    :obj:`float`
        Loss.
    """
    return rmse(results["force"])


def loss_f_e_weighted_mse(results, rho, n_atoms):
    r"""Computes a combined energy and force loss function.

    .. math::

        l = \frac{\rho}{Q} \left\Vert E - \hat{E} \right\Vert^2
        + \frac{1}{n_{atoms} Q} \sum_{i=0}^{n_{atoms}}
        \left\Vert \bf{F}_i - \widehat{\bf{F}}_i \right\Vert^2,

    where :math:`\rho` is a trade-off between energy and force errors,
    :math:`Q` is the number of validation structures, :math:`\Vert \ldots \Vert`
    is the norm, and :math:`\widehat{\;}` is the model prediction of the
    property.

    Parameters
    ----------
    results : :obj:`dict`
        Results which contains at least force and energy errors under the
        ``'force'`` and ``'energy'`` keys, respectively.
    rho : :obj:`float`
        Energy and force trade-off, :math:`\rho`. A recommended value would be
        in the range of ``0.01`` to ``0.1``. Larger values place more
        importance on energy accuracy.
    n_atoms : :obj:`int`
        Number of atoms.

    Returns
    -------
    :obj:`float`
        Loss.
    """
    F_mse = mse(results["force"])
    E_mse = mse(results["energy"])
    return rho * E_mse + (1 / n_atoms) * F_mse
